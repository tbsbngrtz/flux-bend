"""Cross-stream interference — blend weights between image and text streams.

Double-stream blocks have separate weight matrices for image and text streams.
This mode blends them: W' = (1 - alpha) * W_self + alpha * W_other.

Directions:
  img_to_txt:     W_txt' = (1-alpha)*W_txt + alpha*W_img
  txt_to_img:     W_img' = (1-alpha)*W_img + alpha*W_txt
  bidirectional:  both directions simultaneously (reads both originals first)

Attention stream pairs (per double block):
  img to_q    <-> txt add_q_proj
  img to_k    <-> txt add_k_proj
  img to_v    <-> txt add_v_proj
  img to_out.0 <-> txt to_add_out

MLP stream pairs (per double block):
  img ff.linear_in   <-> txt ff_context.linear_in
  img ff.linear_out  <-> txt ff_context.linear_out

Only applies to double blocks (single blocks have no separate streams).
"""

from __future__ import annotations

from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec

# Paired keys: (img_key_suffix, txt_key_suffix)
_ATTN_PAIRS = (
    ("attn.to_q.weight", "attn.add_q_proj.weight"),
    ("attn.to_k.weight", "attn.add_k_proj.weight"),
    ("attn.to_v.weight", "attn.add_v_proj.weight"),
    ("attn.to_out.0.weight", "attn.to_add_out.weight"),
)

_MLP_PAIRS = (
    ("ff.linear_in.weight", "ff_context.linear_in.weight"),
    ("ff.linear_out.weight", "ff_context.linear_out.weight"),
)


class CrossStream(BendingMode):
    name = "cross_stream"
    description = (
        "Blend weights between image and text streams in double-stream blocks. "
        "W' = (1-alpha)*W_self + alpha*W_other."
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "start_block": ParamSpec(
                name="start_block", type="int", required=False, default=0,
                min=0, max=4,
                description="First double block index (inclusive)",
            ),
            "end_block": ParamSpec(
                name="end_block", type="int", required=False, default=-1,
                min=-1, max=4,
                description="Last double block index (inclusive, -1 = last)",
            ),
            "alpha": ParamSpec(
                name="alpha", type="float", required=False, default=0.3,
                min=0.0, max=1.0, step=0.05,
                description="Blend strength: 0 = no change, 1 = full swap",
            ),
            "target": ParamSpec(
                name="target", type="str", required=False, default="attn",
                description='Weight target: "attn", "mlp", or "both"',
            ),
            "direction": ParamSpec(
                name="direction", type="str", required=False, default="bidirectional",
                description='Direction: "img_to_txt", "txt_to_img", or "bidirectional"',
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        target = validated["target"]
        if target not in ("attn", "mlp", "both"):
            raise ValueError(f"target must be 'attn', 'mlp', or 'both', got '{target}'")

        direction = validated["direction"]
        if direction not in ("img_to_txt", "txt_to_img", "bidirectional"):
            raise ValueError(
                f"direction must be 'img_to_txt', 'txt_to_img', or 'bidirectional', "
                f"got '{direction}'"
            )

        if validated["end_block"] == -1:
            validated["end_block"] = model_info.num_double_blocks - 1

        end_block = validated["end_block"]
        if end_block >= model_info.num_double_blocks:
            raise ValueError(
                f"end_block {end_block} exceeds max double block index "
                f"{model_info.num_double_blocks - 1}"
            )

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        start_block = params["start_block"]
        end_block = params["end_block"]
        alpha = params["alpha"]
        target = params["target"]
        direction = params["direction"]

        print(f"  cross_stream: target={target}, direction={direction}, "
              f"alpha={alpha}, blocks=[{start_block}, {end_block}]")

        # Collect pairs to process
        pairs: list[tuple[str, str]] = []
        for i in range(start_block, end_block + 1):
            prefix = f"transformer_blocks.{i}."
            if target in ("attn", "both"):
                for img_suffix, txt_suffix in _ATTN_PAIRS:
                    pairs.append((prefix + img_suffix, prefix + txt_suffix))
            if target in ("mlp", "both"):
                for img_suffix, txt_suffix in _MLP_PAIRS:
                    pairs.append((prefix + img_suffix, prefix + txt_suffix))

        # Sort by img key for deterministic order
        pairs.sort(key=lambda p: p[0])

        modified_count = 0
        for img_key, txt_key in pairs:
            if img_key not in tensors:
                print(f"  WARNING: key '{img_key}' not found, skipping pair")
                continue
            if txt_key not in tensors:
                print(f"  WARNING: key '{txt_key}' not found, skipping pair")
                continue

            img_w = tensors[img_key]
            txt_w = tensors[txt_key]

            if img_w.shape != txt_w.shape:
                print(f"  WARNING: shape mismatch {img_key} {img_w.shape} vs "
                      f"{txt_key} {txt_w.shape}, skipping pair")
                continue

            # Read both originals in float32 BEFORE writing either
            img_f32 = img_w.float()
            txt_f32 = txt_w.float()

            stream_distance = (img_f32 - txt_f32).norm().item()

            if direction == "img_to_txt":
                # Blend img into txt: W_txt' = (1-alpha)*W_txt + alpha*W_img
                txt_bent = (1.0 - alpha) * txt_f32 + alpha * img_f32
                change_norm = (txt_bent - txt_f32).norm().item()

                tensors[txt_key] = txt_bent.to(txt_w.dtype)
                self.check_bf16_survival(txt_key, txt_w, txt_bent)
                print(f"  {img_key} -> {txt_key}: "
                      f"stream_dist={stream_distance:.4f}, change_norm={change_norm:.4f}")
                modified_count += 1

            elif direction == "txt_to_img":
                # Blend txt into img: W_img' = (1-alpha)*W_img + alpha*W_txt
                img_bent = (1.0 - alpha) * img_f32 + alpha * txt_f32
                change_norm = (img_bent - img_f32).norm().item()

                tensors[img_key] = img_bent.to(img_w.dtype)
                self.check_bf16_survival(img_key, img_w, img_bent)
                print(f"  {txt_key} -> {img_key}: "
                      f"stream_dist={stream_distance:.4f}, change_norm={change_norm:.4f}")
                modified_count += 1

            elif direction == "bidirectional":
                # Both directions: compute from originals, then write both
                img_bent = (1.0 - alpha) * img_f32 + alpha * txt_f32
                txt_bent = (1.0 - alpha) * txt_f32 + alpha * img_f32

                img_change = (img_bent - img_f32).norm().item()
                txt_change = (txt_bent - txt_f32).norm().item()

                tensors[img_key] = img_bent.to(img_w.dtype)
                tensors[txt_key] = txt_bent.to(txt_w.dtype)
                self.check_bf16_survival(img_key, img_w, img_bent)
                self.check_bf16_survival(txt_key, txt_w, txt_bent)
                print(f"  {img_key} <-> {txt_key}: "
                      f"stream_dist={stream_distance:.4f}, "
                      f"img_change={img_change:.4f}, txt_change={txt_change:.4f}")
                modified_count += 2

        print(f"  Modified {modified_count} tensor(s)")
