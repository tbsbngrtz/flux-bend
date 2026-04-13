"""Block dropout / zeroing — scale down entire blocks.

Multiplies all weight tensors in selected blocks by `scale` (0.0 = full dropout,
1.0 = no change). Use `target` to restrict to attention or MLP weights only.

Run one block at a time with scale=0.0 to build a block importance map.
This is the recommended first experiment.

Double blocks (indices 0-4) have 16 weight tensors each:
  attn: to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj,
        to_add_out, norm_q, norm_k, norm_added_q, norm_added_k
  mlp:  ff.linear_in, ff.linear_out, ff_context.linear_in, ff_context.linear_out

Single blocks (indices 0-19) have 4 weight tensors each:
  attn+mlp fused: to_qkv_mlp_proj, to_out (contain both attn and MLP)
  attn norms: norm_q, norm_k

When target="attn" or target="mlp" on single blocks, the fused projections
(to_qkv_mlp_proj, to_out) are scaled in full since they contain both streams.
Only the norm weights are exclusively attention.
"""

from __future__ import annotations

from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec

# Double block keys grouped by target
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

# Single block keys grouped by target
_SINGLE_ATTN_SUFFIXES = (
    "attn.norm_q.weight",
    "attn.norm_k.weight",
)

_SINGLE_FUSED_SUFFIXES = (
    "attn.to_qkv_mlp_proj.weight",
    "attn.to_out.weight",
)


class BlockDropout(BendingMode):
    name = "block_dropout"
    description = (
        "Scale down entire blocks (0.0 = full dropout, 1.0 = no change). "
        "Run one block at a time with scale=0.0 to build a block importance map. "
        "This is the recommended first experiment."
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "block_type": ParamSpec(
                name="block_type", type="str", required=False, default="double",
                description='Block type: "double", "single", or "both"',
            ),
            "block_indices": ParamSpec(
                name="block_indices", type="list[int]", required=True, min=0,
                description="Block indices to scale (validated against model architecture)",
            ),
            "scale": ParamSpec(
                name="scale", type="float", required=False, default=0.0,
                min=0.0, max=1.0, step=0.05,
                description="Scale factor: 0.0 = zero out, 1.0 = no change",
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

        # Validate block indices against architecture
        block_indices = validated["block_indices"]
        for idx in block_indices:
            if block_type in ("double", "both"):
                if idx >= model_info.num_double_blocks and block_type == "double":
                    raise ValueError(
                        f"Block index {idx} out of range for double blocks "
                        f"[0, {model_info.num_double_blocks - 1}]"
                    )
            if block_type in ("single", "both"):
                if idx >= model_info.num_single_blocks and block_type == "single":
                    raise ValueError(
                        f"Block index {idx} out of range for single blocks "
                        f"[0, {model_info.num_single_blocks - 1}]"
                    )

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        block_type = params["block_type"]
        block_indices = params["block_indices"]
        scale = params["scale"]
        target = params["target"]

        print(f"  block_dropout: block_type={block_type}, blocks={block_indices}, "
              f"scale={scale}, target={target}")

        modified_count = 0

        # Double blocks
        if block_type in ("double", "both"):
            for idx in sorted(block_indices):
                if idx >= model_info.num_double_blocks:
                    continue

                suffixes: list[str] = []
                if target in ("all", "attn"):
                    suffixes.extend(_DOUBLE_ATTN_SUFFIXES)
                if target in ("all", "mlp"):
                    suffixes.extend(_DOUBLE_MLP_SUFFIXES)

                for suffix in sorted(suffixes):
                    key = f"transformer_blocks.{idx}.{suffix}"
                    modified_count += self._scale_key(tensors, key, scale)

        # Single blocks
        if block_type in ("single", "both"):
            for idx in sorted(block_indices):
                if idx >= model_info.num_single_blocks:
                    continue

                suffixes = []
                if target in ("all", "attn"):
                    suffixes.extend(_SINGLE_ATTN_SUFFIXES)
                if target in ("all", "attn", "mlp"):
                    # Fused projections contain both attn and MLP — include for any target
                    suffixes.extend(_SINGLE_FUSED_SUFFIXES)

                for suffix in sorted(set(suffixes)):
                    key = f"single_transformer_blocks.{idx}.{suffix}"
                    modified_count += self._scale_key(tensors, key, scale)

        print(f"  Modified {modified_count} tensor(s)")

    def _scale_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        scale: float,
    ) -> int:
        """Scale a single tensor. Returns 1 if modified, 0 if skipped."""
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found, skipping")
            return 0

        original = tensors[key]
        w_bent = original.float() * scale

        change_norm = (w_bent - original.float()).norm().item()
        original_norm = original.float().norm().item()

        tensors[key] = w_bent.to(original.dtype)
        self.check_bf16_survival(key, original, w_bent)
        print(f"  {key}: scale={scale}, original_norm={original_norm:.4f}, "
              f"change_norm={change_norm:.4f}")
        return 1
