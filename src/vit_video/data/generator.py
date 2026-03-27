import math
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageOps
import torch

try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

from ..utils.hardware import get_device

class DataGenerator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_with_diffusers(
        self,
        prompts: Dict[str, str],
        num_images_per_class: int = 50,
        device: Optional[torch.device] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> Dict[str, List[Path]]:
        if StableDiffusionPipeline is None:
            raise RuntimeError("diffusers StableDiffusionPipeline is not available.")

        device = device or get_device()
        results: Dict[str, List[Path]] = {}

        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        pipeline = pipeline.to(device)

        rng = random.Random(seed)

        for label, prompt in prompts.items():
            label_dir = self.output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: List[Path] = []
            for i in range(num_images_per_class):
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(
                        rng.randint(0, 2 ** 31 - 1)
                    )

                out = pipeline(
                    prompt,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    num_inference_steps=20,
                )
                image = out.images[0]
                path = label_dir / f"{label}_{i:05d}.png"
                image.save(path)
                saved_paths.append(path)
            results[label] = saved_paths

        return results

    def ensure_existing_examples(
        self, source_examples: Dict[str, List[Path]]
    ) -> Dict[str, List[Path]]:
        results: Dict[str, List[Path]] = {}
        for label, paths in source_examples.items():
            label_dir = self.output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            dests: List[Path] = []
            for i, p in enumerate(paths):
                target = label_dir / f"{label}_{i:05d}{Path(p).suffix}"
                if not Path(p).exists():
                    continue
                if not target.exists():
                    try:
                        Image.open(p).save(target)
                    except Exception:
                        continue
                dests.append(target)
            results[label] = dests
        return results

class VideoProcessor:
    def __init__(self, num_frames: int = 16, out_frame_size: Tuple[int, int] = (224, 224)):
        self.num_frames = num_frames
        self.out_frame_size = out_frame_size

    def _frame_transform(
        self, image: Image.Image, t: float, max_rotation: float = 4.0, max_scale: float = 0.06
    ) -> Image.Image:
        angle = math.sin(2 * math.pi * t) * max_rotation
        scale = 1.0 + math.sin(2 * math.pi * t + math.pi / 4) * max_scale

        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = image.resize((new_w, new_h), Image.BICUBIC)
        img = ImageOps.fit(img, (w, h), method=Image.BICUBIC, centering=(0.5, 0.5))

        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        img = img.resize(self.out_frame_size, Image.BICUBIC)
        return img

    def image_to_video_frames(self, image_path: Path) -> List[Image.Image]:
        with Image.open(image_path) as img_obj:
            img = img_obj.convert("RGB")
        frames: List[Image.Image] = []
        for f in range(self.num_frames):
            t = f / max(1, (self.num_frames - 1))
            frame = self._frame_transform(img, t)
            frames.append(frame)
        return frames

    def save_video_frames(
        self, frames: List[Image.Image], out_dir: Path, prefix: str
    ) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for i, frame in enumerate(frames):
            p = out_dir / f"{prefix}_frame_{i:03d}.png"
            frame.save(p)
            paths.append(p)
        return paths
