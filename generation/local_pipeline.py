"""
generation/local_pipeline.py
═════════════════════════════
Phase 3 — Local SDXL + ControlNet generation on Apple MPS / CUDA.

All three SDXL pipelines share a single set of model weights in memory
(~5GB fp16) to avoid OOM on 24GB unified memory.

Three generation modes:
  text_only  — SDXL text-to-image, no spatial conditioning
  depth      — SDXL + ControlNet conditioned on depth map
  img2img    — SDXL img2img for iterative refinement ("make it warmer")

Usage (called by run_phase3.py, not directly):
  from generation.local_pipeline import LocalSDXLPipeline
  pipe = LocalSDXLPipeline()            # loads once, reuses
  pipe.generate_text_only(...)
  pipe.generate_with_depth(...)
  pipe.refine_img2img(...)
  pipe.unload()                         # free VRAM when done
"""

from __future__ import annotations
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image


DEFAULT_OUTPUTS_DIR = Path("data/outputs/phase3")
DEFAULT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONING_VARIANTS = {
    "depth":      {"use_depth": True, "use_ip_adapter": False},
    # "ip_depth":  {"use_depth": True, "use_ip_adapter": True},  # enabled when IP-Adapter is added
}


def _get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {name}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS GPU")
        return "mps"
    else:
        print("No GPU detected — using CPU (this will be very slow)")
        return "cpu"


class LocalSDXLPipeline:
    """
    Manages SDXL + ControlNet pipelines with shared weights.

    Call load() once — it loads the ControlNet pipeline, then derives
    text-only and img2img pipelines via from_pipe() so they share the
    same UNet/VAE/text-encoder weights. This keeps total VRAM at ~5GB
    instead of ~15GB for three separate loads.
    """

    def __init__(self):
        self.device = _get_device()
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self._pipe_ctrl = None
        self._pipe_txt = None
        self._pipe_i2i = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            ControlNetModel,
            AutoencoderKL,
        )

        t0 = time.time()
        print("Loading SDXL + ControlNet (this takes ~60s on first run)...")

        use_fp16_variant = self.device in ("cuda", "mps")

        controlnet_dtype = torch.float32 if self.device == "mps" else self.dtype
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16" if use_fp16_variant else None,
            use_safetensors=True,
            torch_dtype=controlnet_dtype,
        )

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype,
        )

        self._pipe_ctrl = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16" if use_fp16_variant else None,
            use_safetensors=True,
            torch_dtype=self.dtype,
        )

        self._pipe_txt = StableDiffusionXLPipeline.from_pipe(self._pipe_ctrl)
        self._pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(self._pipe_ctrl)

        if self.device == "cuda":
            self._pipe_ctrl.enable_model_cpu_offload()
            self._pipe_ctrl.enable_vae_slicing()
            self._pipe_ctrl.enable_vae_tiling()
            try:
                self._pipe_ctrl.enable_xformers_memory_efficient_attention()
            except Exception:
                self._pipe_ctrl.enable_attention_slicing()
        elif self.device == "mps":
            self._pipe_ctrl.to("mps")
            self._pipe_ctrl.enable_attention_slicing()
        else:
            self._pipe_ctrl.enable_attention_slicing()

        self._loaded = True
        elapsed = round(time.time() - t0, 1)
        print(f"Models loaded in {elapsed}s on {self.device}")

    def load_lora(self, adapter_path: str, scale: float = 0.8):
        """
        Load a PEFT-trained LoRA adapter onto the UNet.
        Must use PEFT directly — the adapter was saved in PEFT key format,
        not diffusers/kohya format, so load_lora_weights() won't find any keys.
        All three pipelines share the same UNet via from_pipe, so one wrap suffices.
        """
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from peft.tuners.lora import LoraLayer
        from safetensors.torch import load_file

        self.load()
        unet = self._pipe_ctrl.unet

        if hasattr(unet, "base_model"):
            unet.enable_adapter_layers()
        else:
            config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=[
                    "to_q", "to_k", "to_v", "to_out.0",
                    "add_q_proj", "add_k_proj", "add_v_proj",
                ],
                lora_dropout=0.0,
                bias="none",
            )
            unet = get_peft_model(unet, config)
            self._pipe_ctrl.unet = unet
            if self._pipe_txt: self._pipe_txt.unet = unet
            if self._pipe_i2i: self._pipe_i2i.unet = unet

        state_dict = load_file(adapter_path)
        set_peft_model_state_dict(unet, state_dict)

        for module in unet.modules():
            if isinstance(module, LoraLayer):
                for name in module.scaling:
                    module.scaling[name] = scale

        unet.eval()
        print(f"  LoRA loaded: {Path(adapter_path).name}  scale={scale}")

    def unload_lora(self):
        """Disable LoRA layers so subsequent generations use the base model."""
        unet = self._pipe_ctrl.unet
        if hasattr(unet, "base_model"):
            unet.disable_adapter_layers()
            print("  LoRA disabled")

    def unload(self):
        del self._pipe_ctrl, self._pipe_txt, self._pipe_i2i
        self._pipe_ctrl = self._pipe_txt = self._pipe_i2i = None
        self._loaded = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        print("Models unloaded")

    def generate_text_only(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style_name: str,
        seed: int = 42,
        num_steps: int = 30,
        width: int = 768,
        height: int = 768,
        output_dir: str | Path | None = None,
    ) -> str:
        self.load()
        out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        style_slug = style_name.replace(" ", "_")
        out_path = out_dir / f"{style_slug}_text_only.png"

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"  Generating text_only for '{style_name}'...")
        t0 = time.time()

        result = self._pipe_txt(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            generator=generator,
            width=width,
            height=height,
        )

        result.images[0].save(out_path)
        elapsed = round(time.time() - t0, 1)
        print(f"  Saved: {out_path.name} ({elapsed}s)")
        return str(out_path)

    def generate_with_depth(
        self,
        depth_map_path: str,
        positive_prompt: str,
        negative_prompt: str,
        style_name: str,
        conditioning_strength: float = 0.7,
        seed: int = 42,
        num_steps: int = 30,
        width: int = 768,
        height: int = 768,
        output_dir: str | Path | None = None,
    ) -> str:
        self.load()
        out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        style_slug = style_name.replace(" ", "_")
        out_path = out_dir / f"{style_slug}_depth.png"

        depth_image = Image.open(depth_map_path).convert("RGB").resize((width, height))
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"  Generating depth-conditioned for '{style_name}'...")
        t0 = time.time()

        result = self._pipe_ctrl(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=depth_image,
            controlnet_conditioning_scale=conditioning_strength,
            num_inference_steps=num_steps,
            generator=generator,
            width=width,
            height=height,
        )

        result.images[0].save(out_path)
        elapsed = round(time.time() - t0, 1)
        print(f"  Saved: {out_path.name} ({elapsed}s)")
        return str(out_path)

    def refine_img2img(
        self,
        source_image_path: str,
        positive_prompt: str,
        negative_prompt: str,
        refinement_instruction: str = "",
        strength: float = 0.4,
        seed: int = 42,
        num_steps: int = 20,
    ) -> str:
        self.load()
        source = Image.open(source_image_path).convert("RGB")
        out_path = Path(source_image_path).with_stem(
            Path(source_image_path).stem + f"_refined_{seed}"
        )

        if refinement_instruction:
            # prepend so CLIP's 77-token limit hits style tokens last, not first
            positive_prompt = f"{refinement_instruction}, {positive_prompt}"

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"  Running img2img refinement (strength={strength})...")
        t0 = time.time()

        result = self._pipe_i2i(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=source,
            strength=strength,
            num_inference_steps=num_steps,
            generator=generator,
        )

        result.images[0].save(out_path)
        elapsed = round(time.time() - t0, 1)
        print(f"  Saved: {out_path.name} ({elapsed}s)")
        return str(out_path)


# ── Stateless entry points ────────────────────────────────────────────────────

_pipeline_instance: LocalSDXLPipeline | None = None


def _get_pipeline() -> LocalSDXLPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = LocalSDXLPipeline()
    return _pipeline_instance


def request_generation(
    prompts: dict,
    depth_map_path: str,
    seed: int = 42,
    variants: list[str] | None = None,
    output_dir: str | Path | None = None,
    **kwargs,
) -> dict:
    """Generate images for each style x conditioning variant."""
    pipe = _get_pipeline()
    active_variants = set(variants or CONDITIONING_VARIANTS.keys())
    results = {}

    for style, prompt_data in prompts.items():
        results[style] = {}
        positive = prompt_data["positive"]
        negative = prompt_data["negative"]
        strength = prompt_data.get("conditioning_strength", 0.7)

        for variant_name, cond in CONDITIONING_VARIANTS.items():
            if variant_name not in active_variants:
                results[style][variant_name] = None
                continue

            if cond["use_depth"] and (not depth_map_path or not Path(depth_map_path).exists()):
                results[style][variant_name] = None
                continue

            try:
                if cond["use_depth"]:
                    path = pipe.generate_with_depth(
                        depth_map_path, positive, negative, style,
                        conditioning_strength=strength, seed=seed,
                        output_dir=output_dir,
                    )
                else:
                    path = pipe.generate_text_only(
                        positive, negative, style, seed=seed,
                        output_dir=output_dir,
                    )
                results[style][variant_name] = path
            except Exception as e:
                print(f"  Generation failed for {style}/{variant_name}: {e}")
                results[style][variant_name] = None

    return results


def request_img2img(
    source_image_path: str,
    positive_prompt: str,
    negative_prompt: str,
    refinement_instruction: str,
    strength: float = 0.4,
    seed: int = 42,
    **kwargs,
) -> str:
    """Refine an already-generated image with img2img."""
    pipe = _get_pipeline()
    return pipe.refine_img2img(
        source_image_path, positive_prompt, negative_prompt,
        refinement_instruction, strength=strength, seed=seed,
    )
