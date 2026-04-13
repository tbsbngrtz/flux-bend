"""Model loading and tensor I/O for FLUX.2 Klein Base 4B."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def resolve_model_path(model_id_or_path: str) -> Path:
    """Resolve a HuggingFace model ID or local path to a local directory.

    If model_id_or_path contains '/', treats it as a HF repo ID and downloads/caches it.
    Otherwise, validates it as an existing local path.
    """
    if "/" in model_id_or_path:
        local_path = snapshot_download(model_id_or_path)
        return Path(local_path)
    else:
        path = Path(model_id_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return path


def find_transformer_safetensors(model_dir: Path) -> list[Path]:
    """Find all .safetensors files in the transformer/ subfolder."""
    transformer_dir = model_dir / "transformer"
    if not transformer_dir.exists():
        raise FileNotFoundError(
            f"No transformer/ directory found in {model_dir}"
        )
    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {transformer_dir}"
        )
    return files


def load_transformer_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all transformer safetensors shards and merge into one dict.

    All tensors are loaded to CPU.
    """
    files = find_transformer_safetensors(model_dir)
    tensors: dict[str, torch.Tensor] = {}
    for sf_path in files:
        print(f"  Loading {sf_path.name}...")
        shard = load_file(str(sf_path), device="cpu")
        tensors.update(shard)
    print(f"  Loaded {len(tensors)} tensors from {len(files)} file(s)")
    return tensors


def clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Deep-clone all tensors in the dict."""
    return {key: tensor.clone() for key, tensor in tensors.items()}


def save_model(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Save tensors to a safetensors file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved {len(tensors)} tensors to {path} ({size_mb:.1f} MB)")


def remap_keys(
    tensors: dict[str, torch.Tensor], mapping: dict[str, str]
) -> dict[str, torch.Tensor]:
    """Remap tensor keys from raw safetensors names to diffusers names.

    For Klein 4B this is an identity mapping, but kept for API consistency.
    Keys not in the mapping are passed through unchanged.
    """
    if not mapping:
        return tensors
    result: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        new_key = mapping.get(key, key)
        result[new_key] = tensor
    return result


def remap_keys_inverse(
    tensors: dict[str, torch.Tensor], mapping: dict[str, str]
) -> dict[str, torch.Tensor]:
    """Remap tensor keys from diffusers names back to raw safetensors names.

    For Klein 4B this is an identity mapping, but kept for API consistency.
    """
    if not mapping:
        return tensors
    inverse = {v: k for k, v in mapping.items()}
    return remap_keys(tensors, inverse)
