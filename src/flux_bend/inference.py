"""Pipeline loading and image generation for FLUX.2 Klein Base 4B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def load_pipeline(model_id_or_path: str) -> Any:
    """Load the Flux2KleinPipeline and enable CPU offloading.

    The pipeline is loaded once and reused for all jobs.
    Returns the pipeline object.
    """
    from diffusers import Flux2KleinPipeline

    print(f"Loading pipeline from {model_id_or_path}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("Pipeline loaded with model_cpu_offload enabled.")
    return pipe


def patch_pipeline(
    pipeline: Any,
    bent_tensors: dict[str, torch.Tensor],
    key_mapping: dict[str, str],
) -> None:
    """Patch the pipeline's transformer with bent tensors.

    For Klein 4B, key_mapping is empty (identity), so bent_tensors keys
    are already in diffusers format.
    """
    if key_mapping:
        from flux_bend.loader import remap_keys
        remapped = remap_keys(bent_tensors, key_mapping)
    else:
        remapped = bent_tensors

    state_dict = {
        k: v.to(torch.bfloat16) for k, v in remapped.items()
    }

    pipeline.transformer.load_state_dict(state_dict, strict=True)
    print(f"  Patched transformer with {len(state_dict)} tensors")


def generate_image(
    pipeline: Any,
    prompt: str,
    seed: int,
    steps: int = 50,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
) -> Image.Image:
    """Generate a single image with deterministic seeding.

    Seeds both CPU and CUDA RNGs. Uses CUDA generator for latent noise.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
        generator=generator,
    )
    image = result.images[0]

    arr = np.array(image)
    print(f"  Image stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

    return image


def load_source_image(image_path: Path, width: int, height: int) -> Image.Image:
    """Load a source image for img2img, resizing if dimensions don't match.

    Returns the PIL Image in RGB mode, resized to (width, height) if needed.
    """
    img = Image.open(image_path).convert("RGB")
    if img.size != (width, height):
        print(f"  WARNING: Source image {img.size[0]}x{img.size[1]} "
              f"doesn't match target {width}x{height}, resizing")
        img = img.resize((width, height), Image.LANCZOS)
    return img


def generate_image_i2i(
    pipeline: Any,
    image: Image.Image,
    prompt: str,
    seed: int,
    steps: int = 50,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
) -> Image.Image:
    """Generate an image-to-image edit with deterministic seeding.

    The source image is passed directly to the pipeline which handles
    VAE encoding internally. Seeds both CPU and CUDA RNGs.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipeline(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
        generator=generator,
    )
    out_image = result.images[0]

    arr = np.array(out_image)
    print(f"  Image stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

    return out_image
