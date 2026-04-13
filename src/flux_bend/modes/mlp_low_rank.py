"""MLP low-rank noise injection — add structured noise to MLP weights."""

from __future__ import annotations

import math
from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec


class MlpLowRank(BendingMode):
    name = "mlp_low_rank"
    description = "Add deterministic low-rank noise N = (U @ V) / sqrt(d_in) to MLP weights"

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "block_type": ParamSpec(
                name="block_type", type="str", required=False, default="single",
                description='Block type: "double", "single", or "both"',
            ),
            "stream": ParamSpec(
                name="stream", type="str", required=False, default="both",
                description='MLP stream: "img", "txt", or "both" (double blocks only)',
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
            "rank": ParamSpec(
                name="rank", type="int", required=False, default=8,
                min=1, max=64, step=1,
                description="Number of noise directions (rank of U @ V)",
            ),
            "alpha": ParamSpec(
                name="alpha", type="float", required=False, default=2.5,
                min=0.1, max=10.0, step=0.1,
                description="Noise strength multiplier",
            ),
            "seed": ParamSpec(
                name="seed", type="int", required=True,
                description="RNG seed for noise matrices (REQUIRED for determinism)",
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
            else:  # both
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
        stream = params["stream"]
        target_weight = params["target_weight"]
        start_block = params["start_block"]
        end_block = params["end_block"]
        rank = params["rank"]
        alpha = params["alpha"]
        seed = params["seed"]

        hidden_size = model_info.hidden_size
        mlp_hidden_dim = model_info.mlp_hidden_dim

        # Seed ONCE at start — iteration order determines RNG consumption
        torch.manual_seed(seed)

        print(f"  mlp_low_rank: block_type={block_type}, stream={stream}, "
              f"target_weight={target_weight}, blocks=[{start_block}, {end_block}], "
              f"rank={rank}, alpha={alpha}, seed={seed}")

        # Collect all target keys with their noise shapes and slice info.
        # Each entry: (key, noise_rows, noise_cols, slice_spec)
        # slice_spec is None for non-fused tensors, or a descriptor for fused ones.
        targets: list[tuple[str, int, int, str | None]] = []

        # Double blocks
        if block_type in ("double", "both"):
            db_end = min(end_block, model_info.num_double_blocks - 1)
            db_start = min(start_block, db_end)

            for i in range(db_start, db_end + 1):
                if stream in ("img", "both"):
                    if target_weight in ("linear_in", "all"):
                        key = f"transformer_blocks.{i}.ff.linear_in.weight"
                        # Shape: (2 * mlp_hidden_dim, hidden_size) = (18432, 3072)
                        targets.append((key, 2 * mlp_hidden_dim, hidden_size, None))
                    if target_weight in ("linear_out", "all"):
                        key = f"transformer_blocks.{i}.ff.linear_out.weight"
                        # Shape: (hidden_size, mlp_hidden_dim) = (3072, 9216)
                        targets.append((key, hidden_size, mlp_hidden_dim, None))

                if stream in ("txt", "both"):
                    if target_weight in ("linear_in", "all"):
                        key = f"transformer_blocks.{i}.ff_context.linear_in.weight"
                        targets.append((key, 2 * mlp_hidden_dim, hidden_size, None))
                    if target_weight in ("linear_out", "all"):
                        key = f"transformer_blocks.{i}.ff_context.linear_out.weight"
                        targets.append((key, hidden_size, mlp_hidden_dim, None))

        # Single blocks (fused projections — only modify MLP slices)
        if block_type in ("single", "both"):
            sb_end = min(end_block, model_info.num_single_blocks - 1)
            sb_start = min(start_block, sb_end)

            for i in range(sb_start, sb_end + 1):
                if target_weight in ("linear_in", "all"):
                    key = f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight"
                    # MLP slice: rows [hidden_size*3 : hidden_size*3 + 2*mlp_hidden_dim]
                    # = rows [9216:27648], noise shape (18432, 3072)
                    targets.append((key, 2 * mlp_hidden_dim, hidden_size, "fused_input"))
                if target_weight in ("linear_out", "all"):
                    key = f"single_transformer_blocks.{i}.attn.to_out.weight"
                    # MLP slice: cols [hidden_size : hidden_size + mlp_hidden_dim]
                    # = cols [3072:12288], noise shape (3072, 9216)
                    targets.append((key, hidden_size, mlp_hidden_dim, "fused_output"))

        # Sort alphabetically for deterministic RNG consumption order
        targets.sort(key=lambda t: t[0])

        print(f"  Sorted iteration order ({len(targets)} targets):")
        for key, _, _, _ in targets:
            print(f"    {key}")

        modified_count = 0
        for key, noise_rows, noise_cols, slice_spec in targets:
            modified_count += self._inject_noise_to_key(
                tensors, key, noise_rows, noise_cols, slice_spec,
                rank, alpha, hidden_size, mlp_hidden_dim,
            )

        print(f"  Modified {modified_count} tensor(s)")

    def _inject_noise_to_key(
        self,
        tensors: dict[str, torch.Tensor],
        key: str,
        noise_rows: int,
        noise_cols: int,
        slice_spec: str | None,
        rank: int,
        alpha: float,
        hidden_size: int,
        mlp_hidden_dim: int,
    ) -> int:
        """Inject low-rank noise into one tensor. Returns 1 if modified, 0 if skipped."""
        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            # Still consume RNG to keep deterministic order
            torch.randn(noise_rows, rank)
            torch.randn(rank, noise_cols)
            return 0

        original = tensors[key]

        # Generate noise in float32
        U = torch.randn(noise_rows, rank)
        V = torch.randn(rank, noise_cols)
        N = (U @ V) / math.sqrt(noise_cols)

        noise_norm = (alpha * N).norm().item()

        if slice_spec is None:
            # Non-fused: noise applies to entire tensor
            w = original.float()
            w_bent = w + alpha * N
            change_norm = (w_bent - w).norm().item()

            tensors[key] = w_bent.to(original.dtype)
            self.check_bf16_survival(key, original, w_bent)

        elif slice_spec == "fused_input":
            # Fused QKV+MLP input: only modify MLP rows [hidden_size*3:]
            w = original.float().clone()
            qkv_dim = hidden_size * 3  # 9216
            mlp_slice = w[qkv_dim:, :]
            mlp_slice_bent = mlp_slice + alpha * N
            w[qkv_dim:, :] = mlp_slice_bent
            change_norm = (w - original.float()).norm().item()

            tensors[key] = w.to(original.dtype)
            self.check_bf16_survival(key, original, w)

        elif slice_spec == "fused_output":
            # Fused attn+MLP output: only modify MLP cols [hidden_size:]
            w = original.float().clone()
            mlp_cols = w[:, hidden_size:]
            mlp_cols_bent = mlp_cols + alpha * N
            w[:, hidden_size:] = mlp_cols_bent
            change_norm = (w - original.float()).norm().item()

            tensors[key] = w.to(original.dtype)
            self.check_bf16_survival(key, original, w)

        else:
            raise ValueError(f"Unknown slice_spec: {slice_spec}")

        print(f"  {key}: rank={rank}, alpha={alpha}, "
              f"noise_norm={noise_norm:.4f}, change_norm={change_norm:.4f}"
              + (f" ({slice_spec})" if slice_spec else ""))
        return 1
