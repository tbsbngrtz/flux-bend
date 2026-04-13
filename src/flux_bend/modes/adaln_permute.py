"""AdaLN modulation permutation — reorder shared modulation groups.

Shared AdaLN-Zero modulation weights produce shift/scale/gate signals for all
blocks of each type. The output dimension is divided into equal groups of
hidden_size (3072).

Double-stream modulation (img and txt each):
  6 groups of 3072, output dim = 18432
  Group 0: shift_attn  — additive shift before attention
  Group 1: scale_attn  — multiplicative scale before attention
  Group 2: gate_attn   — post-attention gating
  Group 3: shift_ff    — additive shift before feed-forward
  Group 4: scale_ff    — multiplicative scale before feed-forward
  Group 5: gate_ff     — post-feed-forward gating

  Keys: double_stream_modulation_img.linear.weight  (18432, 3072)
        double_stream_modulation_txt.linear.weight  (18432, 3072)

Single-stream modulation:
  3 groups of 3072, output dim = 9216
  Group 0: shift  — additive shift
  Group 1: scale  — multiplicative scale
  Group 2: gate   — post-block gating

  Key: single_stream_modulation.linear.weight  (9216, 3072)

Permuting these groups scrambles what signal each position encodes,
e.g. swapping shift and gate makes the model gate where it should shift.
"""

from __future__ import annotations

from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec

# Human-readable labels for each group position.
_DOUBLE_GROUP_LABELS = (
    "shift_attn",
    "scale_attn",
    "gate_attn",
    "shift_ff",
    "scale_ff",
    "gate_ff",
)

_SINGLE_GROUP_LABELS = (
    "shift",
    "scale",
    "gate",
)


def _parse_permutation(perm_str: str) -> list[int]:
    """Parse '2-0-1-5-4-3' into [2, 0, 1, 5, 4, 3]."""
    return [int(x) for x in perm_str.strip().split("-")]


def _validate_permutation(perm: list[int], expected_len: int, target_name: str) -> None:
    """Check that perm is a valid permutation of [0, expected_len)."""
    if len(perm) != expected_len:
        raise ValueError(
            f"Permutation for {target_name} must have {expected_len} elements, "
            f"got {len(perm)}: {perm}"
        )
    if sorted(perm) != list(range(expected_len)):
        raise ValueError(
            f"Permutation for {target_name} must be a rearrangement of "
            f"{list(range(expected_len))}, got {perm}"
        )


def _permute_groups(
    weight: torch.Tensor,
    perm: list[int],
    group_size: int,
) -> torch.Tensor:
    """Reorder output-dimension groups of a modulation weight.

    weight shape: (num_groups * group_size, in_dim)
    Returns a new tensor with groups reordered according to perm.
    """
    w = weight.float()
    num_groups = len(perm)
    in_dim = w.shape[1]

    # Split into groups along output dim
    groups = [w[g * group_size : (g + 1) * group_size, :] for g in range(num_groups)]

    # Reassemble in permuted order
    permuted = torch.cat([groups[perm[i]] for i in range(num_groups)], dim=0)
    return permuted


class AdalNPermute(BendingMode):
    name = "adaln_permute"
    description = (
        "Permute shared AdaLN-Zero modulation groups. "
        "Double: 6 groups (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff). "
        "Single: 3 groups (shift, scale, gate)."
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "target": ParamSpec(
                name="target", type="str", required=False, default="double",
                description='Target: "double", "single", or "both"',
            ),
            "permutation": ParamSpec(
                name="permutation", type="str", required=True,
                description=(
                    'Group permutation as dash-separated indices. '
                    'Double: 6 elements (e.g. "2-0-1-5-4-3"). '
                    'Single: 3 elements (e.g. "2-0-1"). '
                    'When target="both", provide double permutation only; '
                    'single permutation is derived by mapping groups modulo 3.'
                ),
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        target = validated["target"]
        if target not in ("double", "single", "both"):
            raise ValueError(f"target must be 'double', 'single', or 'both', got '{target}'")

        perm = _parse_permutation(validated["permutation"])

        if target == "double":
            _validate_permutation(perm, 6, "double modulation")
        elif target == "single":
            _validate_permutation(perm, 3, "single modulation")
        else:  # both
            _validate_permutation(perm, 6, "double modulation (target='both')")

        # Store parsed permutation back as the raw string (validated)
        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        target = params["target"]
        perm = _parse_permutation(params["permutation"])
        hidden_size = model_info.hidden_size

        print(f"  adaln_permute: target={target}, permutation={perm}")

        modified_count = 0

        # Double-stream modulation (img and txt)
        if target in ("double", "both"):
            double_perm = perm if target != "both" else perm
            # both always uses the 6-element perm for double

            print(f"  Double modulation: 6 groups of {hidden_size}")
            print(f"    Original order: {list(range(6))} = {list(_DOUBLE_GROUP_LABELS)}")
            new_labels = [_DOUBLE_GROUP_LABELS[double_perm[i]] for i in range(6)]
            print(f"    New order:      {double_perm} = {new_labels}")
            print(f"    Meaning: position 0 now carries '{new_labels[0]}' "
                  f"(was '{_DOUBLE_GROUP_LABELS[0]}')")

            for key in (
                "double_stream_modulation_img.linear.weight",
                "double_stream_modulation_txt.linear.weight",
            ):
                if key not in tensors:
                    print(f"  WARNING: key '{key}' not found in tensors, skipping")
                    continue

                original = tensors[key]
                permuted_f32 = _permute_groups(original, double_perm, hidden_size)
                change_norm = (permuted_f32 - original.float()).norm().item()

                tensors[key] = permuted_f32.to(original.dtype)
                self.check_bf16_survival(key, original, permuted_f32)
                print(f"  {key}: change_norm={change_norm:.4f}")
                modified_count += 1

        # Single-stream modulation
        if target in ("single", "both"):
            if target == "both":
                # Derive 3-element permutation from 6-element:
                # Map double groups to single: 0,1,2 (attn shift/scale/gate) -> 0,1,2 (shift/scale/gate)
                # Groups 3,4,5 in double are ff shift/scale/gate — modulo 3 gives 0,1,2
                # Take the first 3 positions of the double perm, modulo 3
                single_perm = [perm[i] % 3 for i in range(3)]
                # Validate it's still a valid permutation
                if sorted(single_perm) != [0, 1, 2]:
                    print(f"  WARNING: derived single permutation {single_perm} is not valid, "
                          f"using identity [0, 1, 2] for single blocks")
                    single_perm = [0, 1, 2]
            else:
                single_perm = perm

            print(f"  Single modulation: 3 groups of {hidden_size}")
            print(f"    Original order: {list(range(3))} = {list(_SINGLE_GROUP_LABELS)}")
            new_labels = [_SINGLE_GROUP_LABELS[single_perm[i]] for i in range(3)]
            print(f"    New order:      {single_perm} = {new_labels}")
            print(f"    Meaning: position 0 now carries '{new_labels[0]}' "
                  f"(was '{_SINGLE_GROUP_LABELS[0]}')")

            key = "single_stream_modulation.linear.weight"
            if key not in tensors:
                print(f"  WARNING: key '{key}' not found in tensors, skipping")
            else:
                original = tensors[key]
                permuted_f32 = _permute_groups(original, single_perm, hidden_size)
                change_norm = (permuted_f32 - original.float()).norm().item()

                tensors[key] = permuted_f32.to(original.dtype)
                self.check_bf16_survival(key, original, permuted_f32)
                print(f"  {key}: change_norm={change_norm:.4f}")
                modified_count += 1

        print(f"  Modified {modified_count} tensor(s)")
