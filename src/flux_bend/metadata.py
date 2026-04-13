"""PNG metadata embedding and tensor checksum utilities.

Provides:
- save_image_with_metadata(): save PIL Image with tEXt chunks for reproducibility
- compute_tensor_checksums(): sha256[:8] of modified tensors after bending
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from flux_bend import __version__


def build_png_metadata(
    *,
    model: str,
    mode: str,
    params: dict[str, Any] | list[dict[str, Any]],
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    tensor_checksums: dict[str, str] | None = None,
    source_image_hash: str | None = None,
) -> PngInfo:
    """Build a PngInfo object with reproducibility metadata as tEXt chunks.

    All values are stored as strings in standard PNG tEXt chunks.
    """
    info = PngInfo()
    info.add_text("flux-bend:model", model)
    info.add_text("flux-bend:mode", mode)
    info.add_text("flux-bend:params", json.dumps(params, default=str))
    info.add_text("flux-bend:prompt", prompt)
    info.add_text("flux-bend:seed", str(seed))
    info.add_text("flux-bend:steps", str(steps))
    info.add_text("flux-bend:guidance_scale", str(guidance_scale))
    info.add_text("flux-bend:width", str(width))
    info.add_text("flux-bend:height", str(height))
    info.add_text("flux-bend:timestamp", datetime.now(timezone.utc).isoformat())
    info.add_text("flux-bend:tool_version", __version__)
    if tensor_checksums:
        info.add_text("flux-bend:tensor_checksums", json.dumps(tensor_checksums))
    if source_image_hash:
        info.add_text("flux-bend:source_image_hash", source_image_hash)
    return info


def save_image_with_metadata(
    image: Image.Image,
    path: Path,
    *,
    model: str,
    mode: str,
    params: dict[str, Any] | list[dict[str, Any]],
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    tensor_checksums: dict[str, str] | None = None,
    source_image_hash: str | None = None,
) -> None:
    """Save a PIL Image as PNG with embedded tEXt metadata chunks."""
    png_info = build_png_metadata(
        model=model,
        mode=mode,
        params=params,
        prompt=prompt,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        tensor_checksums=tensor_checksums,
        source_image_hash=source_image_hash,
    )
    image.save(path, pnginfo=png_info)


def compute_tensor_checksums(
    original_tensors: dict[str, torch.Tensor],
    bent_tensors: dict[str, torch.Tensor],
) -> dict[str, str]:
    """Compute sha256[:8] checksums for tensors that were modified by bending.

    Compares bent_tensors against original_tensors. Only includes keys
    where the tensor bytes differ. Returns {key: sha256_hex[:8]}.
    """
    checksums: dict[str, str] = {}
    for key in sorted(bent_tensors.keys()):
        bent = bent_tensors[key]
        if key in original_tensors:
            orig = original_tensors[key]
            # Quick check: same shape and dtype, then compare bytes
            if orig.shape == bent.shape and orig.dtype == bent.dtype:
                if torch.equal(orig, bent):
                    continue
        # Tensor was modified — compute checksum of bent tensor bytes
        h = hashlib.sha256(bent.contiguous().cpu().to(torch.float32).numpy().tobytes()).hexdigest()[:8]
        checksums[key] = h
    return checksums
