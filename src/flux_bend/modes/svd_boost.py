"""SVD component boost — scale specific singular values of weight matrices.

Uses torch.pca_lowrank to compute a partial SVD (only the top-q components),
then scales selected singular values and applies the resulting delta to the
original weight. This preserves all other singular values exactly.

Algorithm per tensor:
  torch.manual_seed(seed)  # pca_lowrank uses random initialization
  q = max(components) + 1
  U, S, V = pca_lowrank(W, q=q)
  S_new[i] = S[i] * scales[i]  for each i in components
  delta = U @ diag(S_new - S) @ V^T
  W' = W + delta
"""

from __future__ import annotations

import time
from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec


class SvdBoost(BendingMode):
    name = "svd_boost"
    description = "Scale specific singular values of weight matrices via partial SVD"

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
            "target_weight": ParamSpec(
                name="target_weight", type="str", required=False, default="all",
                description='Which MLP weight: "linear_in", "linear_out", or "all"',
            ),
            "start_block": ParamSpec(
                name="start_block", type="int", required=False, default=0, min=0,
                description="First block index (inclusive)",
            ),
            "end_block": ParamSpec(
                name="end_block", type="int", required=False, default=-1, min=-1,
                description="Last block index (inclusive, -1 = last)",
            ),
            "components": ParamSpec(
                name="components", type="list[int]", required=True, min=0,
                description="0-indexed singular value indices to modify (0 = largest SV)",
            ),
            "scales": ParamSpec(
                name="scales", type="str", required=True,
                description=(
                    "Scale factors for each component, dash-separated (e.g. '2.0-0.5-1.5'). "
                    "Must have same length as components. Each SV is multiplied by its scale."
                ),
            ),
            "seed": ParamSpec(
                name="seed", type="int", required=True,
                description="RNG seed for pca_lowrank (REQUIRED for determinism)",
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

        target_weight = validated["target_weight"]
        if target_weight not in ("linear_in", "linear_out", "all"):
            raise ValueError(f"target_weight must be 'linear_in', 'linear_out', or 'all', got '{target_weight}'")

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

        # Parse scales from dash-separated string
        scales_str = validated["scales"]
        scales = [float(x) for x in scales_str.strip().split("-")]
        validated["_scales_parsed"] = scales

        components = validated["components"]
        if len(scales) != len(components):
            raise ValueError(
                f"scales has {len(scales)} values but components has {len(components)} — "
                f"must be same length"
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
        target_weight = params["target_weight"]
        start_block = params["start_block"]
        end_block = params["end_block"]
        components = params["components"]
        scales = params["_scales_parsed"]
        seed = params["seed"]

        hidden_size = model_info.hidden_size
        mlp_hidden_dim = model_info.mlp_hidden_dim

        q = max(components) + 1

        print(f"  svd_boost: block_type={block_type}, stream={stream}, "
              f"target_weight={target_weight}, blocks=[{start_block}, {end_block}], "
              f"components={components}, scales={scales}, q={q}, seed={seed}")

        # Collect targets: (key, slice_spec)
        # slice_spec: None for non-fused, "fused_input" or "fused_output" for single blocks
        targets: list[tuple[str, str | None]] = []

        # Double blocks
        if block_type in ("double", "both"):
            db_end = min(end_block, model_info.num_double_blocks - 1)
            db_start = min(start_block, db_end)

            for i in range(db_start, db_end + 1):
                if stream in ("img", "both"):
                    if target_weight in ("linear_in", "all"):
                        targets.append((f"transformer_blocks.{i}.ff.linear_in.weight", None))
                    if target_weight in ("linear_out", "all"):
                        targets.append((f"transformer_blocks.{i}.ff.linear_out.weight", None))
                if stream in ("txt", "both"):
                    if target_weight in ("linear_in", "all"):
                        targets.append((f"transformer_blocks.{i}.ff_context.linear_in.weight", None))
                    if target_weight in ("linear_out", "all"):
                        targets.append((f"transformer_blocks.{i}.ff_context.linear_out.weight", None))

        # Single blocks (fused projections — only MLP slices)
        if block_type in ("single", "both"):
            sb_end = min(end_block, model_info.num_single_blocks - 1)
            sb_start = min(start_block, sb_end)

            for i in range(sb_start, sb_end + 1):
                if target_weight in ("linear_in", "all"):
                    targets.append((
                        f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight",
                        "fused_input",
                    ))
                if target_weight in ("linear_out", "all"):
                    targets.append((
                        f"single_transformer_blocks.{i}.attn.to_out.weight",
                        "fused_output",
                    ))

        # Sort alphabetically for deterministic iteration
        targets.sort(key=lambda t: t[0])

        print(f"  Sorted iteration order ({len(targets)} targets):")
        for key, _ in targets:
            print(f"    {key}")

        modified_count = 0
        for key, slice_spec in targets:
            modified_count += self._boost_key(
                tensors, key, slice_spec,
                components, scales, q, seed,
                hidden_size, mlp_hidden_dim,
            )

        print(f"  Modified {modified_count} tensor(s)")

    def _boost_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        slice_spec: str | None,
        components: list[int],
        scales: list[float],
        q: int,
        seed: int,
        hidden_size: int,
        mlp_hidden_dim: int,
    ) -> int:
        """Apply SVD boost to one tensor. Returns 1 if modified, 0 if skipped."""
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return 0

        original = tensors[key]
        t0 = time.perf_counter()

        if slice_spec is None:
            # Non-fused: SVD on entire tensor
            w = original.float()
            w_bent = self._svd_boost_matrix(w, components, scales, q, seed)
            change_norm = (w_bent - w).norm().item()
            tensors[key] = w_bent.to(original.dtype)
            self.check_bf16_survival(key, original, w_bent)

        elif slice_spec == "fused_input":
            # Fused QKV+MLP input: only modify MLP rows [hidden_size*3:]
            w_full = original.float().clone()
            qkv_dim = hidden_size * 3
            mlp_slice = w_full[qkv_dim:, :]
            mlp_bent = self._svd_boost_matrix(mlp_slice, components, scales, q, seed)
            w_full[qkv_dim:, :] = mlp_bent
            change_norm = (w_full - original.float()).norm().item()
            tensors[key] = w_full.to(original.dtype)
            self.check_bf16_survival(key, original, w_full)

        elif slice_spec == "fused_output":
            # Fused attn+MLP output: only modify MLP cols [hidden_size:]
            w_full = original.float().clone()
            mlp_cols = w_full[:, hidden_size:]
            mlp_bent = self._svd_boost_matrix(mlp_cols, components, scales, q, seed)
            w_full[:, hidden_size:] = mlp_bent
            change_norm = (w_full - original.float()).norm().item()
            tensors[key] = w_full.to(original.dtype)
            self.check_bf16_survival(key, original, w_full)

        else:
            raise ValueError(f"Unknown slice_spec: {slice_spec}")

        elapsed = time.perf_counter() - t0
        print(f"  {key}: change_norm={change_norm:.4f}, time={elapsed:.2f}s"
              + (f" ({slice_spec})" if slice_spec else ""))
        return 1

    @staticmethod
    def _svd_boost_matrix(
        w: torch.Tensor,
        components: list[int],
        scales: list[float],
        q: int,
        seed: int,
    ) -> torch.Tensor:
        """Compute partial SVD, scale selected singular values, return modified matrix.

        Uses pca_lowrank which requires a seed for its random initialization.
        The delta is: U @ diag(S_new - S) @ V^T, added to the original weight.
        """
        torch.manual_seed(seed)

        U, S, V = torch.pca_lowrank(w, q=q)
        # U: (m, q), S: (q,), V: (n, q)

        S_new = S.clone()
        for idx, scale in zip(components, scales):
            original_sv = S[idx].item()
            S_new[idx] = S[idx] * scale
            new_sv = S_new[idx].item()
            print(f"    SV[{idx}]: {original_sv:.4f} -> {new_sv:.4f} (x{scale})")

        # Compute delta: U @ diag(S_new - S) @ V^T
        delta_S = S_new - S
        delta = U @ torch.diag(delta_S) @ V.T

        return w + delta
