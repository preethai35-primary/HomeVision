"""
lora/trainer.py
═══════════════
LoRA fine-tuning for SDXL on curated interior style images.

Trains a lightweight LoRA adapter (~50 MB) on 20-50 images of a target
interior style. The adapter is activated at inference by a trigger token
(e.g. "IVI" for Indian Vintage) prepended to the style prompt.

Targets UNet cross-attention layers only (q/k/v/out projections).
Text encoders are frozen — style is injected via captions.

Usage:
  python lora/trainer.py \
    --images  data/lora/indian_vintage/ \
    --style   "Indian Vintage" \
    --trigger IVI \
    --output  lora/adapters/indian_vintage.safetensors \
    --steps   1000

Hardware estimate:
  Apple M2 Pro  — ~2.5-3 h for 1000 steps at 1024x1024
  NVIDIA 8 GB   — ~40 min for 1000 steps
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# ensure project root is on sys.path so `lora.dataset` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LORA_RANK       = 8
LORA_ALPHA      = 16      # typically 2x rank
LORA_DROPOUT    = 0.05
LEARNING_RATE   = 1e-4
LR_WARMUP_STEPS = 50
SAVE_EVERY      = 250
BATCH_SIZE      = 1       # keep at 1 for MPS / 8 GB VRAM

UNET_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj",
]


def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_models(device: str):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    dtype    = torch.float16 if device in ("cuda", "mps") else torch.float32
    use_fp16 = device in ("cuda", "mps")
    base     = "stabilityai/stable-diffusion-xl-base-1.0"

    print("  Loading tokenizers...")
    tok1 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")

    print("  Loading text encoders (frozen)...")
    te1 = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(
        base, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    te1.requires_grad_(False)
    te2.requires_grad_(False)

    print("  Loading VAE (frozen)...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)

    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        base,
        subfolder="unet",
        torch_dtype=dtype,
        variant="fp16" if use_fp16 else None,
        use_safetensors=True,
    ).to(device)

    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    return tok1, tok2, te1, te2, vae, unet, scheduler


def _apply_lora(unet):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=UNET_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    unet = get_peft_model(unet, config)
    unet.print_trainable_parameters()
    return unet


def _encode_text(ids1, ids2, te1, te2, device):
    import torch
    with torch.no_grad():
        out1 = te1(ids1.to(device), output_hidden_states=True)
        embeds1 = out1.hidden_states[-2]
        out2 = te2(ids2.to(device), output_hidden_states=True)
        pooled  = out2[0]
        embeds2 = out2.hidden_states[-2]
    return torch.cat([embeds1, embeds2], dim=-1), pooled


def _save(unet, path: Path, style: str, trigger: str, step: int):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    state = get_peft_model_state_dict(unet)
    save_file(state, str(path), metadata={
        "style": style, "trigger": trigger,
        "step": str(step), "rank": str(LORA_RANK),
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
    })
    print(f"  [saved] {path.name}")


def _plot_loss(losses: list[float], output_path: Path):
    try:
        import matplotlib.pyplot as plt
        smoothed = []
        window = 20
        for i in range(len(losses)):
            start = max(0, i - window)
            smoothed.append(sum(losses[start:i+1]) / (i - start + 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(losses,   color="#d0d0d0", linewidth=0.8, label="raw loss")
        ax.plot(smoothed, color="#e05c2a", linewidth=2.0, label=f"smoothed (window={window})")
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE loss")
        ax.set_title(f"LoRA training loss — {output_path.stem}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = output_path.with_suffix(".loss.png")
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  [loss curve] saved to {plot_path}")
    except ImportError:
        print("  (install matplotlib to get loss curve PNG)")


def train_lora(
    image_dir: str,
    output_path: str,
    style_name: str,
    trigger: str,
    steps: int = 1000,
):
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    from torch.utils.data import DataLoader
    from lora.dataset import LoRADataset

    device = _get_device()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype  = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"\n[lora] device={device}  style={style_name!r}  trigger={trigger!r}  steps={steps}")
    print("[lora] Loading models (uses HF cache — no re-download)...")
    tok1, tok2, te1, te2, vae, unet, scheduler = _load_models(device)

    # gradient checkpointing: recomputes activations during backward instead of
    # storing them — cuts peak MPS memory ~50% at cost of ~30% more compute
    unet.enable_gradient_checkpointing()

    unet = _apply_lora(unet)
    unet.train()

    dataset = LoRADataset(
        image_dir=image_dir,
        trigger=trigger,
        style_name=style_name,
        tokenizer_1=tok1,
        tokenizer_2=tok2,
        target_size=1024,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # pre-encode all captions once, then offload text encoders to CPU —
    # they're frozen and don't need GPU memory during the UNet training loop
    print("[lora] Pre-encoding captions...")
    all_embeds: list[tuple] = []
    with torch.no_grad():
        for item in dataset:
            e, p = _encode_text(
                item["input_ids_1"].unsqueeze(0),
                item["input_ids_2"].unsqueeze(0),
                te1, te2, device,
            )
            all_embeds.append((e.cpu(), p.cpu()))

    te1.to("cpu")
    te2.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()
    print(f"[lora] Text encoders offloaded — {torch.mps.current_allocated_memory() / 1e9:.1f} GB on MPS\n"
          if device == "mps" else "[lora] Text encoders offloaded to CPU\n")

    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=LR_WARMUP_STEPS
    )

    print(f"[lora] Training — checkpoints every {SAVE_EVERY} steps\n")
    t0     = time.time()
    step   = 0
    losses = []
    indices = list(range(len(dataset)))

    while step < steps:
        import random
        random.shuffle(indices)
        for idx in indices:
            if step >= steps:
                break

            item         = dataset[idx]
            pixel_values = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise      = torch.randn_like(latents)
            timesteps  = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (1,), device=device,
            ).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, pooled = all_embeds[idx]
            prompt_embeds = prompt_embeds.to(device, dtype=dtype)
            pooled        = pooled.to(device, dtype=dtype)

            bs = latents.shape[0]
            time_ids = torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]] * bs, dtype=dtype, device=device
            )

            noise_pred = unet(
                noisy,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype),
                added_cond_kwargs={
                    "text_embeds": pooled.to(dtype),
                    "time_ids":    time_ids,
                },
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if step < LR_WARMUP_STEPS:
                warmup_scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 50 == 0:
                avg     = sum(losses[-50:]) / 50
                elapsed = (time.time() - t0) / 60
                eta     = (elapsed / step) * (steps - step)
                print(f"  step {step:4d}/{steps}  loss={avg:.4f}  "
                      f"elapsed={elapsed:.1f}m  eta={eta:.1f}m")

            if step % SAVE_EVERY == 0 and step < steps:
                ckpt = output.with_stem(f"{output.stem}_step{step}")
                _save(unet, ckpt, style_name, trigger, step)

    _save(unet, output, style_name, trigger, steps)
    _plot_loss(losses, output)

    total = (time.time() - t0) / 60
    print(f"\n[lora] Done in {total:.1f} min")
    print(f"[lora] Adapter: {output}")
    print(f"[lora] Trigger: prepend '{trigger},' to any prompt to activate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  required=True)
    parser.add_argument("--style",   required=True)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--steps",   type=int, default=1000)
    args = parser.parse_args()

    train_lora(
        image_dir=args.images,
        output_path=args.output,
        style_name=args.style,
        trigger=args.trigger,
        steps=args.steps,
    )
