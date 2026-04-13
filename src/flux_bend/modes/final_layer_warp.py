"""Final layer warp — modify the final projection from hidden to latent space.

Target tensor: proj_out.weight (128, 3072)
This is the last linear layer that maps from model hidden dim (3072) to
latent patch space (128 channels). Modifying it directly affects the
decoded image output.

Three sub-modes via `target`:

target = "scale":
  Uniform scaling: W' = W * scale.
  No seed needed.

target = "channel_permute":
  Permute output channels (rows of proj_out.weight).
  The 128 output channels map to VAE latent channels.
  Permuting them scrambles which latent channel receives which hidden
  representation, producing color/structure distortions.
  Accepts shorthand: "reverse", "shift-N", "swap-A-B", or explicit list.
  No seed needed.

target = "low_rank_noise":
  Add low-rank noise N = (U @ V) / sqrt(d_in) to proj_out.weight.
  Small alpha recommended (<0.5) since this is the final projection
  and changes here directly affect every pixel.
  Seed REQUIRED.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec

_PROJ_OUT_KEY = "proj_out.weight"


def _parse_permutation_spec(spec: str | list[int], num_channels: int) -> list[int]:
    """Parse a permutation specification into an explicit index list.

    Accepts:
      - list[int]: explicit full permutation (validated)
      - "reverse": reverse order
      - "shift-N": circular shift by N positions
      - "swap-A-B": swap channels A and B, leave rest unchanged
    """
    if isinstance(spec, list):
        return spec

    spec = str(spec).strip()

    if spec == "reverse":
        return list(range(num_channels - 1, -1, -1))

    if spec.startswith("shift-"):
        try:
            n = int(spec[6:])
        except ValueError:
            raise ValueError(f"Invalid shift amount in '{spec}', expected 'shift-N' with integer N")
        return [(i - n) % num_channels for i in range(num_channels)]

    if spec.startswith("swap-"):
        parts = spec[5:].split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid swap spec '{spec}', expected 'swap-A-B'")
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid swap indices in '{spec}', expected integers")
        if a < 0 or a >= num_channels:
            raise ValueError(f"Swap index {a} out of range [0, {num_channels - 1}]")
        if b < 0 or b >= num_channels:
            raise ValueError(f"Swap index {b} out of range [0, {num_channels - 1}]")
        perm = list(range(num_channels))
        perm[a], perm[b] = perm[b], perm[a]
        return perm

    raise ValueError(
        f"Unknown permutation spec '{spec}'. "
        f"Expected: list[int], 'reverse', 'shift-N', or 'swap-A-B'"
    )


def _validate_permutation(perm: list[int], num_channels: int) -> None:
    """Check that perm is a valid permutation of [0, num_channels)."""
    if len(perm) != num_channels:
        raise ValueError(
            f"Permutation must have {num_channels} elements, got {len(perm)}"
        )
    if sorted(perm) != list(range(num_channels)):
        raise ValueError(
            f"Permutation must be a rearrangement of [0, {num_channels - 1}], "
            f"got {perm[:20]}{'...' if len(perm) > 20 else ''}"
        )


class FinalLayerWarp(BendingMode):
    name = "final_layer_warp"
    description = (
        "Modify the final projection (proj_out.weight, 128x3072). "
        "Sub-modes: 'scale' (uniform), 'channel_permute' (reorder output channels), "
        "'low_rank_noise' (add structured noise)."
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "target": ParamSpec(
                name="target", type="str", required=True,
                description='Sub-mode: "scale", "channel_permute", or "low_rank_noise"',
            ),
            "scale": ParamSpec(
                name="scale", type="float", required=False, default=1.0,
                min=0.1, max=5.0, step=0.1,
                description="Uniform scale factor (scale mode only)",
            ),
            "permutation": ParamSpec(
                name="permutation", type="str", required=False, default="reverse",
                description=(
                    'Channel permutation spec (channel_permute mode only). '
                    'Options: "reverse", "shift-N", "swap-A-B", or explicit list'
                ),
            ),
            "rank": ParamSpec(
                name="rank", type="int", required=False, default=4,
                min=1, max=64, step=1,
                description="Noise rank (low_rank_noise mode only)",
            ),
            "alpha": ParamSpec(
                name="alpha", type="float", required=False, default=0.1,
                min=0.001, max=5.0, step=0.01,
                description="Noise strength (low_rank_noise mode only, <0.5 recommended)",
            ),
            "seed": ParamSpec(
                name="seed", type="int", required=False, default=None,
                description="RNG seed (REQUIRED for low_rank_noise mode)",
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        target = validated["target"]
        if target not in ("scale", "channel_permute", "low_rank_noise"):
            raise ValueError(
                f"target must be 'scale', 'channel_permute', or 'low_rank_noise', "
                f"got '{target}'"
            )

        if target == "low_rank_noise" and validated["seed"] is None:
            raise ValueError("seed is REQUIRED for low_rank_noise mode")

        if target == "channel_permute":
            num_channels = model_info.in_channels  # 128
            perm = _parse_permutation_spec(validated["permutation"], num_channels)
            _validate_permutation(perm, num_channels)
            validated["_permutation_parsed"] = perm

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        target = params["target"]

        if target == "scale":
            self._apply_scale(tensors, params)
        elif target == "channel_permute":
            self._apply_channel_permute(tensors, params, model_info)
        elif target == "low_rank_noise":
            self._apply_low_rank_noise(tensors, params)

    def _apply_scale(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
    ) -> None:
        """Uniform scaling: W' = W * scale."""
        scale = params["scale"]
        print(f"  final_layer_warp(scale): scale={scale}")

        if _PROJ_OUT_KEY not in tensors:
            print(f"  WARNING: key '{_PROJ_OUT_KEY}' not found, skipping")
            return

        original = tensors[_PROJ_OUT_KEY]
        w_bent = original.float() * scale

        change_norm = (w_bent - original.float()).norm().item()
        tensors[_PROJ_OUT_KEY] = w_bent.to(original.dtype)
        self.check_bf16_survival(_PROJ_OUT_KEY, original, w_bent)
        print(f"  {_PROJ_OUT_KEY}: scale={scale}, change_norm={change_norm:.4f}")
        print(f"  Modified 1 tensor(s)")

    def _apply_channel_permute(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        """Permute output channels (rows of proj_out.weight)."""
        perm = params["_permutation_parsed"]
        spec = params["permutation"]
        num_channels = model_info.in_channels

        print(f"  final_layer_warp(channel_permute): spec='{spec}', "
              f"num_channels={num_channels}")

        if _PROJ_OUT_KEY not in tensors:
            print(f"  WARNING: key '{_PROJ_OUT_KEY}' not found, skipping")
            return

        original = tensors[_PROJ_OUT_KEY]
        w = original.float()

        # Permute rows (output channels)
        w_permuted = w[perm, :]

        change_norm = (w_permuted - w).norm().item()
        num_moved = sum(1 for i, p in enumerate(perm) if i != p)

        tensors[_PROJ_OUT_KEY] = w_permuted.to(original.dtype)
        self.check_bf16_survival(_PROJ_OUT_KEY, original, w_permuted)
        print(f"  {_PROJ_OUT_KEY}: {num_moved}/{num_channels} channels moved, "
              f"change_norm={change_norm:.4f}")
        print(f"  Permutation (first 16): {perm[:16]}...")
        print(f"  Modified 1 tensor(s)")

    def _apply_low_rank_noise(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
    ) -> None:
        """Add low-rank noise to proj_out.weight."""
        rank = params["rank"]
        alpha = params["alpha"]
        seed = params["seed"]

        print(f"  final_layer_warp(low_rank_noise): rank={rank}, alpha={alpha}, seed={seed}")

        if _PROJ_OUT_KEY not in tensors:
            print(f"  WARNING: key '{_PROJ_OUT_KEY}' not found, skipping")
            return

        torch.manual_seed(seed)

        original = tensors[_PROJ_OUT_KEY]
        w = original.float()
        d_out, d_in = w.shape

        U = torch.randn(d_out, rank)
        V = torch.randn(rank, d_in)
        N = (U @ V) / math.sqrt(d_in)

        w_bent = w + alpha * N

        noise_norm = (alpha * N).norm().item()
        change_norm = (w_bent - w).norm().item()
        weight_norm = w.norm().item()

        tensors[_PROJ_OUT_KEY] = w_bent.to(original.dtype)
        self.check_bf16_survival(_PROJ_OUT_KEY, original, w_bent)
        print(f"  {_PROJ_OUT_KEY}: rank={rank}, alpha={alpha}, "
              f"noise_norm={noise_norm:.6f}, change_norm={change_norm:.6f}, "
              f"weight_norm={weight_norm:.4f}, noise/weight={noise_norm/weight_norm:.6e}")
        print(f"  Modified 1 tensor(s)")
