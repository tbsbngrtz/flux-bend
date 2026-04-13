"""Attention head scaling mode — scale selected heads by (1 + strength)."""

from __future__ import annotations

from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec


def _scale_qkv_weight(
    weight: torch.Tensor,
    heads: list[int],
    num_heads: int,
    head_dim: int,
    strength: float,
) -> torch.Tensor:
    """Scale selected head rows in a Q/K/V projection weight.

    Weight shape: (num_heads * head_dim, in_features).
    Output dim encodes heads, so reshape as (num_heads, head_dim, in_features).
    """
    w = weight.float()
    in_features = w.shape[1]
    w = w.view(num_heads, head_dim, in_features)
    scale = 1.0 + strength
    for h in heads:
        w[h] *= scale
    return w.view(weight.shape)


def _scale_output_weight(
    weight: torch.Tensor,
    heads: list[int],
    num_heads: int,
    head_dim: int,
    strength: float,
) -> torch.Tensor:
    """Scale selected head columns in an output projection weight.

    Weight shape: (out_features, num_heads * head_dim).
    Input dim encodes heads, so reshape as (out_features, num_heads, head_dim).
    """
    w = weight.float()
    out_features = w.shape[0]
    w = w.view(out_features, num_heads, head_dim)
    scale = 1.0 + strength
    for h in heads:
        w[:, h, :] *= scale
    return w.view(weight.shape)


class AttnHeadScale(BendingMode):
    name = "attn_head_scale"
    description = "Scale attention head weights by (1 + strength) for selected heads"

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "block_type": ParamSpec(
                name="block_type", type="str", required=False, default="double",
                description='Block type: "double", "single", or "both"',
            ),
            "stream": ParamSpec(
                name="stream", type="str", required=False, default="img",
                description='Stream: "img", "txt", or "both" (double blocks only)',
            ),
            "start_block": ParamSpec(
                name="start_block", type="int", required=False, default=0, min=0,
                description="First block index (inclusive)",
            ),
            "end_block": ParamSpec(
                name="end_block", type="int", required=False, default=-1, min=-1,
                description="Last block index (inclusive, -1 = last)",
            ),
            "heads": ParamSpec(
                name="heads", type="list[int]", required=True, min=0, max=23,
                description="Head indices to scale",
            ),
            "strength": ParamSpec(
                name="strength", type="float", required=False, default=0.5,
                min=-0.95, max=3.0, step=0.05,
                description="Scale factor: each head multiplied by (1 + strength)",
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        block_type = validated["block_type"]
        if block_type not in ("double", "single", "both"):
            raise ValueError(f"block_type must be 'double', 'single', or 'both', got '{block_type}'")

        stream = validated["stream"]
        if stream not in ("img", "txt", "both"):
            raise ValueError(f"stream must be 'img', 'txt', or 'both', got '{stream}'")

        if block_type == "single" and stream == "txt":
            raise ValueError("Single blocks have no separate text stream (use stream='img' or 'both')")

        # Resolve end_block=-1
        if validated["end_block"] == -1:
            if block_type == "double":
                validated["end_block"] = model_info.num_double_blocks - 1
            elif block_type == "single":
                validated["end_block"] = model_info.num_single_blocks - 1
            else:  # both
                validated["end_block"] = max(
                    model_info.num_double_blocks - 1,
                    model_info.num_single_blocks - 1,
                )

        # Validate head indices against actual num_heads
        for h in validated["heads"]:
            if h < 0 or h >= model_info.num_heads:
                raise ValueError(
                    f"Head index {h} out of range [0, {model_info.num_heads - 1}]"
                )

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        block_type = params["block_type"]
        stream = params["stream"]
        start_block = params["start_block"]
        end_block = params["end_block"]
        heads = params["heads"]
        strength = params["strength"]
        num_heads = model_info.num_heads
        head_dim = model_info.head_dim
        hidden_size = model_info.hidden_size

        assert hidden_size == num_heads * head_dim, (
            f"hidden_size ({hidden_size}) != num_heads ({num_heads}) * head_dim ({head_dim})"
        )

        print(f"  attn_head_scale: block_type={block_type}, stream={stream}, "
              f"blocks=[{start_block}, {end_block}], heads={heads}, strength={strength}")

        modified_count = 0

        # Double blocks
        if block_type in ("double", "both"):
            db_end = min(end_block, model_info.num_double_blocks - 1)
            db_start = min(start_block, db_end)

            for i in range(db_start, db_end + 1):
                # Image stream
                if stream in ("img", "both"):
                    for suffix in ("to_q", "to_k", "to_v"):
                        key = f"transformer_blocks.{i}.attn.{suffix}.weight"
                        modified_count += self._scale_qkv_key(
                            tensors, key, heads, num_heads, head_dim, strength
                        )
                    # Output projection
                    key = f"transformer_blocks.{i}.attn.to_out.0.weight"
                    modified_count += self._scale_output_key(
                        tensors, key, heads, num_heads, head_dim, strength
                    )

                # Text stream
                if stream in ("txt", "both"):
                    for suffix in ("add_q_proj", "add_k_proj", "add_v_proj"):
                        key = f"transformer_blocks.{i}.attn.{suffix}.weight"
                        modified_count += self._scale_qkv_key(
                            tensors, key, heads, num_heads, head_dim, strength
                        )
                    key = f"transformer_blocks.{i}.attn.to_add_out.weight"
                    modified_count += self._scale_output_key(
                        tensors, key, heads, num_heads, head_dim, strength
                    )

        # Single blocks (fused projections)
        if block_type in ("single", "both"):
            sb_end = min(end_block, model_info.num_single_blocks - 1)
            sb_start = min(start_block, sb_end)

            for i in range(sb_start, sb_end + 1):
                # Fused QKV+MLP input
                key = f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight"
                modified_count += self._scale_fused_qkv_key(
                    tensors, key, heads, num_heads, head_dim, hidden_size, strength
                )

                # Fused attention+MLP output
                key = f"single_transformer_blocks.{i}.attn.to_out.weight"
                modified_count += self._scale_fused_output_key(
                    tensors, key, heads, num_heads, head_dim, hidden_size, strength
                )

        print(f"  Modified {modified_count} tensor(s)")

    def _scale_qkv_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        heads: list[int],
        num_heads: int,
        head_dim: int,
        strength: float,
    ) -> int:
        """Scale heads in a separate Q/K/V projection. Returns 1 if modified, 0 if key missing."""
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return 0

        original = tensors[key]
        bent_f32 = _scale_qkv_weight(original, heads, num_heads, head_dim, strength)
        change_norm = (bent_f32 - original.float()).norm().item()

        tensors[key] = bent_f32.to(original.dtype)
        self.check_bf16_survival(key, original, bent_f32)
        print(f"  {key}: heads={heads}, strength={strength}, change_norm={change_norm:.4f}")
        return 1

    def _scale_output_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        heads: list[int],
        num_heads: int,
        head_dim: int,
        strength: float,
    ) -> int:
        """Scale heads in an output projection. Returns 1 if modified, 0 if key missing."""
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return 0

        original = tensors[key]
        bent_f32 = _scale_output_weight(original, heads, num_heads, head_dim, strength)
        change_norm = (bent_f32 - original.float()).norm().item()

        tensors[key] = bent_f32.to(original.dtype)
        self.check_bf16_survival(key, original, bent_f32)
        print(f"  {key}: heads={heads}, strength={strength}, change_norm={change_norm:.4f}")
        return 1

    def _scale_fused_qkv_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        heads: list[int],
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        strength: float,
    ) -> int:
        """Scale heads in a fused QKV+MLP projection (single blocks).

        Only modifies the QKV rows (first 3*hidden_size), leaves MLP rows untouched.
        """
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return 0

        original = tensors[key]
        w = original.float().clone()
        qkv_dim = hidden_size * 3  # Q + K + V

        # Extract Q, K, V slices (each hidden_size rows)
        for offset_name, start in [("Q", 0), ("K", hidden_size), ("V", hidden_size * 2)]:
            end = start + hidden_size
            qkv_slice = w[start:end, :]
            scaled = _scale_qkv_weight(qkv_slice, heads, num_heads, head_dim, strength)
            w[start:end, :] = scaled

        change_norm = (w - original.float()).norm().item()
        tensors[key] = w.to(original.dtype)
        self.check_bf16_survival(key, original, w)
        print(f"  {key}: heads={heads}, strength={strength}, change_norm={change_norm:.4f} (fused QKV)")
        return 1

    def _scale_fused_output_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        heads: list[int],
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        strength: float,
    ) -> int:
        """Scale heads in a fused attention+MLP output projection (single blocks).

        Only modifies the attention output columns (first hidden_size), leaves MLP columns untouched.
        """
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return 0

        original = tensors[key]
        w = original.float().clone()

        # Extract attention output columns
        attn_cols = w[:, :hidden_size]
        scaled = _scale_output_weight(attn_cols, heads, num_heads, head_dim, strength)
        w[:, :hidden_size] = scaled

        change_norm = (w - original.float()).norm().item()
        tensors[key] = w.to(original.dtype)
        self.check_bf16_survival(key, original, w)
        print(f"  {key}: heads={heads}, strength={strength}, change_norm={change_norm:.4f} (fused output)")
        return 1
