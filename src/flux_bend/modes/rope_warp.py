"""RoPE warp — manipulate spatial position perception.

EXCEPTION TO WEIGHTS-ONLY RULE: RoPE frequencies in Flux2 are computed at
runtime by the pos_embed module (Flux2PosEmbed), not stored in weights.
This mode uses an inference hook to scale position IDs before RoPE computation.

The pos_embed module receives position IDs of shape (S, 4) with axes:
  axis 0: T (time, always 0 for image tokens)
  axis 1: H (height position, 0..H-1)
  axis 2: W (width position, 0..W-1)
  axis 3: L (layer, always 0 for image tokens)

Each axis gets its own set of RoPE frequencies with axes_dims_rope[i] dimensions.
Scaling the H/W position IDs changes the effective spatial frequency:
  freq_scale > 1: positions spread apart -> model "sees" larger image -> zoomed-in effect
  freq_scale < 1: positions compressed -> model "sees" smaller image -> zoomed-out effect

The hook registers on pipeline.transformer.pos_embed and scales the input
position IDs before the RoPE frequency computation. It is installed before
inference and cleaned up after, leaving the model unchanged.

No weights are modified by this mode (apply() is a no-op).
"""

from __future__ import annotations

from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec


class RopeWarp(BendingMode):
    name = "rope_warp"
    description = (
        "Scale RoPE position frequencies via inference hook (not weight modification). "
        "Warps spatial perception: >1 zooms in, <1 zooms out. "
        "Axes: 'all' scales H+W, 'height'/'width' scales one axis only."
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "freq_scale": ParamSpec(
                name="freq_scale", type="float", required=False, default=1.0,
                min=0.1, max=10.0, step=0.1,
                description="Position ID scale factor (1.0 = no change)",
            ),
            "axis": ParamSpec(
                name="axis", type="str", required=False, default="all",
                description='Which spatial axis: "all" (H+W), "height", or "width"',
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        axis = validated["axis"]
        if axis not in ("all", "height", "width"):
            raise ValueError(f"axis must be 'all', 'height', or 'width', got '{axis}'")

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        """No-op: RoPE is computed at runtime, not stored in weights."""
        freq_scale = params["freq_scale"]
        axis = params["axis"]
        print(f"  rope_warp: freq_scale={freq_scale}, axis={axis}")
        print(f"  NOTE: This mode uses an inference hook, not weight modification.")
        print(f"  No tensors modified. Hook will be applied during inference.")

    def get_inference_hooks(
        self,
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> list[Any]:
        """Return a hook that scales position IDs before RoPE computation."""
        freq_scale = params["freq_scale"]
        axis = params["axis"]

        if freq_scale == 1.0:
            print(f"  rope_warp: freq_scale=1.0, no hook needed")
            return []

        def install_hook(pipeline: Any) -> Any:
            """Install forward pre-hook on pos_embed. Returns cleanup function."""
            pos_embed = pipeline.transformer.pos_embed

            # Position ID axes: [T=0, H=1, W=2, L=3]
            scale_axes: list[int] = []
            if axis in ("all", "height"):
                scale_axes.append(1)  # H
            if axis in ("all", "width"):
                scale_axes.append(2)  # W

            def pre_hook(module: Any, args: tuple) -> tuple:
                """Scale position IDs before RoPE frequency computation."""
                ids = args[0]  # shape: (S, 4)
                ids_scaled = ids.clone()
                for ax in scale_axes:
                    ids_scaled[..., ax] = ids_scaled[..., ax] * freq_scale
                return (ids_scaled,)

            handle = pos_embed.register_forward_pre_hook(pre_hook)

            print(f"  rope_warp hook installed: freq_scale={freq_scale}, "
                  f"axis={axis}, scaling axes {scale_axes}")

            def cleanup() -> None:
                handle.remove()
                print(f"  rope_warp hook removed")

            return cleanup

        return [install_hook]
