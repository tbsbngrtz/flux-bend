"""Weight quantization noise — round weights to fewer effective bits.

Simulates low-bit quantization by snapping each weight to a reduced grid.
The quantization is deterministic (no seed needed) — same weights always
produce the same quantized output.

Algorithm per tensor:
  1. Find value range: vmin, vmax = w.min(), w.max()
  2. Compute grid: num_levels = 2^bits, step = (vmax - vmin) / (num_levels - 1)
  3. Quantize: w_q = round((w - vmin) / step) * step + vmin
  4. Result is still in float32/bf16 but only uses 2^bits distinct values per tensor.

Lower bits = more aggressive quantization = more glitch:
  8 bits: 256 levels — subtle noise, nearly lossless
  4 bits: 16 levels  — visible degradation
  2 bits: 4 levels   — heavy quantization
  1 bit:  2 levels   — binary weights (extreme)
"""

from __future__ import annotations

from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec

# Reuse key groupings from block_dropout
_DOUBLE_ATTN_SUFFIXES = (
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "attn.add_q_proj.weight",
    "attn.add_k_proj.weight",
    "attn.add_v_proj.weight",
    "attn.to_add_out.weight",
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.norm_added_q.weight",
    "attn.norm_added_k.weight",
)

_DOUBLE_MLP_SUFFIXES = (
    "ff.linear_in.weight",
    "ff.linear_out.weight",
    "ff_context.linear_in.weight",
    "ff_context.linear_out.weight",
)

_SINGLE_ATTN_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
)

_SINGLE_FUSED_SUFFIXES = (
    "attn.to_qkv_mlp_proj.weight",
    "attn.to_out.weight",
)


class QuantNoise(BendingMode):
    name = "quant_noise"
    description = (
        "Round weights to fewer effective bits (deterministic quantization). "
        "Lower bits = more aggressive rounding = more glitch."
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "block_type": ParamSpec(
                name="block_type", type="str", required=False, default="double",
                description='Block type: "double", "single", or "both"',
            ),
            "start_block": ParamSpec(
                name="start_block", type="int", required=False, default=0, min=0,
                description="First block index (inclusive)",
            ),
            "end_block": ParamSpec(
                name="end_block", type="int", required=False, default=-1, min=-1,
                description="Last block index (inclusive, -1 = last)",
            ),
            "bits": ParamSpec(
                name="bits", type="int", required=False, default=4,
                min=1, max=8, step=1,
                description="Number of quantization bits (1=binary, 8=nearly lossless)",
            ),
            "target": ParamSpec(
                name="target", type="str", required=False, default="all",
                description='Weight target: "all", "attn", or "mlp"',
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        block_type = validated["block_type"]
        if block_type not in ("double", "single", "both"):
            raise ValueError(f"block_type must be 'double', 'single', or 'both', got '{block_type}'")

        target = validated["target"]
        if target not in ("all", "attn", "mlp"):
            raise ValueError(f"target must be 'all', 'attn', or 'mlp', got '{target}'")

        # Resolve end_block=-1
        if validated["end_block"] == -1:
            if block_type == "double":
                validated["end_block"] = model_info.num_double_blocks - 1
            elif block_type == "single":
                validated["end_block"] = model_info.num_single_blocks - 1
            else:
                validated["end_block"] = max(
                    model_info.num_double_blocks - 1,
                    model_info.num_single_blocks - 1,
                )

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        block_type = params["block_type"]
        start_block = params["start_block"]
        end_block = params["end_block"]
        bits = params["bits"]
        target = params["target"]
        num_levels = 2 ** bits

        print(f"  quant_noise: block_type={block_type}, blocks=[{start_block}, {end_block}], "
              f"bits={bits}, levels={num_levels}, target={target}")

        # Collect keys
        keys: list[str] = []

        if block_type in ("double", "both"):
            db_end = min(end_block, model_info.num_double_blocks - 1)
            db_start = min(start_block, db_end)
            for i in range(db_start, db_end + 1):
                suffixes: list[str] = []
                if target in ("all", "attn"):
                    suffixes.extend(_DOUBLE_ATTN_SUFFIXES)
                if target in ("all", "mlp"):
                    suffixes.extend(_DOUBLE_MLP_SUFFIXES)
                for suffix in suffixes:
                    keys.append(f"transformer_blocks.{i}.{suffix}")

        if block_type in ("single", "both"):
            sb_end = min(end_block, model_info.num_single_blocks - 1)
            sb_start = min(start_block, sb_end)
            for i in range(sb_start, sb_end + 1):
                suffixes = []
                if target in ("all", "attn"):
                    suffixes.extend(_SINGLE_ATTN_SUFFIXES)
                if target in ("all", "attn", "mlp"):
                    suffixes.extend(_SINGLE_FUSED_SUFFIXES)
                for suffix in set(suffixes):
                    keys.append(f"single_transformer_blocks.{i}.{suffix}")

        keys.sort()

        modified_count = 0
        for key in keys:
            modified_count += self._quantize_key(tensors, key, num_levels)

        print(f"  Modified {modified_count} tensor(s)")

    def _quantize_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        num_levels: int,
    ) -> int:
        """Quantize a single tensor to num_levels. Returns 1 if modified, 0 if skipped."""
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found, skipping")
            return 0

        original = tensors[key]
        w = original.float()

        vmin = w.min()
        vmax = w.max()
        value_range = vmax - vmin

        if value_range == 0:
            print(f"  {key}: constant tensor, skipping")
            return 0

        step = value_range / (num_levels - 1)

        # Quantize: snap to nearest grid point
        w_quantized = torch.round((w - vmin) / step) * step + vmin

        # Compute statistics
        error = (w_quantized - w).abs()
        quant_error_norm = error.norm().item()
        max_error = error.max().item()
        mean_error = error.mean().item()
        num_changed = (w_quantized != w).sum().item()
        total_weights = w.numel()
        pct_changed = 100.0 * num_changed / total_weights

        tensors[key] = w_quantized.to(original.dtype)
        self.check_bf16_survival(key, original, w_quantized)
        print(f"  {key}: quant_error_norm={quant_error_norm:.4f}, "
              f"max_error={max_error:.6f}, mean_error={mean_error:.6f}, "
              f"changed={num_changed}/{total_weights} ({pct_changed:.1f}%)")
        return 1
