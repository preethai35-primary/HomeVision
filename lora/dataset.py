"""
lora/dataset.py
═══════════════
Dataset preparation for SDXL LoRA fine-tuning.

Drop 20-50 curated interior images in a folder — this module handles
resizing, augmentation, and auto-captioning via GPT-4o-mini.

Captions are cached in <image_dir>/captions.json after the first run
so you're not billed twice.
"""

from __future__ import annotations
import base64
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _media_type(path: Path) -> str:
    return "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"


def generate_captions(
    image_dir: Path,
    trigger: str,
    style_name: str,
    force: bool = False,
) -> dict[str, str]:
    """
    Caption all images using GPT-4o-mini. Results cached to captions.json.
    Each caption starts with the trigger token so the LoRA learns to
    activate on it.
    """
    cache_path = image_dir / "captions.json"
    images     = [p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]

    if cache_path.exists() and not force:
        captions    = json.loads(cache_path.read_text())
        uncaptioned = [p for p in images if p.name not in captions]
        if not uncaptioned:
            print(f"[dataset] captions loaded from cache ({len(captions)} images)")
            return captions
    else:
        captions    = {}
        uncaptioned = images

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()

    system = (
        f"You are captioning interior design images to fine-tune a diffusion model "
        f"on {style_name} style. Write ONE sentence describing the image — focus on "
        f"materials, furniture, colours, textures, lighting, atmosphere. "
        f"Do NOT mention photo quality or camera. "
        f"Start with '{trigger},' (the trigger token, with comma)."
    )

    for i, path in enumerate(uncaptioned):
        print(f"  Captioning {i+1}/{len(uncaptioned)}: {path.name}")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=120,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/{_media_type(path)};base64,{_encode_image(path)}",
                        "detail": "low",
                    }},
                    {"type": "text", "text": system},
                ]}],
            )
            captions[path.name] = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Warning: {path.name} — {e}")
            captions[path.name] = f"{trigger}, {style_name} interior design"

    cache_path.write_text(json.dumps(captions, indent=2, ensure_ascii=False))
    print(f"[dataset] captions saved → {cache_path}")
    return captions


class LoRADataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        trigger: str,
        style_name: str,
        tokenizer_1,
        tokenizer_2,
        target_size: int = 1024,
        augment: bool = True,
    ):
        self.image_dir   = Path(image_dir)
        self.trigger     = trigger
        self.style_name  = style_name
        self.tok1        = tokenizer_1
        self.tok2        = tokenizer_2
        self.target_size = target_size
        self.augment     = augment

        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        )
        if not self.images:
            raise ValueError(f"No images found in {image_dir}")

        print(f"[dataset] {len(self.images)} images")
        self.captions = generate_captions(self.image_dir, trigger, style_name)

        print(f"\n{'─'*60}")
        for fname, caption in sorted(self.captions.items()):
            print(f"  {fname}\n    {caption}\n")
        print(f"{'─'*60}")
        print("Review captions above. Edit data/lora/.../captions.json to fix any.")
        print("Training starts in 5 seconds (Ctrl+C to abort)...\n")
        import time; time.sleep(5)

    def __len__(self):
        return len(self.images)

    def _load(self, path: Path) -> torch.Tensor:
        img  = Image.open(path).convert("RGB")
        w, h = img.size
        s    = min(w, h)
        img  = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        img  = img.resize((self.target_size, self.target_size), Image.LANCZOS)

        if self.augment and torch.rand(1).item() > 0.5:
            import torchvision.transforms.functional as TF
            img = TF.hflip(img)

        arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
        arr = arr.reshape(self.target_size, self.target_size, 3)
        return arr.permute(2, 0, 1) / 127.5 - 1.0

    def _tok(self, text: str, tokenizer):
        return tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        path    = self.images[idx]
        caption = self.captions.get(path.name, f"{self.trigger}, {self.style_name} interior")
        return {
            "pixel_values": self._load(path),
            "input_ids_1":  self._tok(caption, self.tok1),
            "input_ids_2":  self._tok(caption, self.tok2),
            "caption":      caption,
        }
