"""Embedding space interventions — warp timestep or latent channel embeddings.

Two sub-modes selected via `target`:

target = "timestep":
  Add low-rank noise to the timestep embedder's final linear layer
  (time_guidance_embed.timestep_embedder.linear_2.weight, shape 3072x3072).
  This distorts the model's sense of noise level across diffusion steps.
  N = (U @ V) / sqrt(d_in), W' = W + alpha * N.
  Uses small alpha (default 0.01) — bf16 survival check is critical.

target = "latent_channel":
  Apply a near-orthogonal rotation to the latent input projection
  (x_embedder.weight, shape 3072x128).
  M = (1 - alpha) * I + alpha * Q, where Q is an orthogonal matrix from QR
  decomposition of a seeded random matrix. W' = M @ W.
  This mixes latent channels while approximately preserving norms.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from flux_bend.model_info import ModelInfo
from flux_bend.modes._base import BendingMode, ParamSpec


class EmbedWarp(BendingMode):
    name = "embed_warp"
    description = (
        "Warp embedding spaces: 'timestep' adds low-rank noise to timestep embedder, "
        "'latent_channel' applies near-orthogonal rotation to latent input projection"
    )

    @classmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        return {
            "target": ParamSpec(
                name="target", type="str", required=True,
                description='Sub-mode: "timestep" or "latent_channel"',
            ),
            "rank": ParamSpec(
                name="rank", type="int", required=False, default=4,
                min=1, max=64, step=1,
                description="Noise rank (timestep mode only)",
            ),
            "alpha": ParamSpec(
                name="alpha", type="float", required=False, default=0.01,
                min=0.001, max=10.0, step=0.001,
                description=(
                    "Strength. timestep: noise multiplier (default 0.01, small values!). "
                    "latent_channel: interpolation toward orthogonal rotation (default 0.3, max 1.0)."
                ),
            ),
            "seed": ParamSpec(
                name="seed", type="int", required=True,
                description="RNG seed (REQUIRED for determinism)",
            ),
        }

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        validated = super().validate(params, model_info)

        target = validated["target"]
        if target not in ("timestep", "latent_channel"):
            raise ValueError(f"target must be 'timestep' or 'latent_channel', got '{target}'")

        # Apply target-specific alpha defaults and bounds
        if target == "latent_channel":
            alpha = validated["alpha"]
            # If user didn't provide alpha explicitly and it's still the timestep default,
            # use latent_channel default
            if "alpha" not in params:
                validated["alpha"] = 0.3
            elif alpha > 1.0:
                raise ValueError(
                    f"alpha for latent_channel must be <= 1.0, got {alpha}"
                )

        return validated

    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        target = params["target"]
        seed = params["seed"]

        torch.manual_seed(seed)

        if target == "timestep":
            self._apply_timestep(tensors, params)
        elif target == "latent_channel":
            self._apply_latent_channel(tensors, params, model_info)

    def _apply_timestep(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
    ) -> None:
        """Add low-rank noise to timestep embedder final linear."""
        rank = params["rank"]
        alpha = params["alpha"]

        key = "time_guidance_embed.timestep_embedder.linear_2.weight"

        print(f"  embed_warp(timestep): rank={rank}, alpha={alpha}, seed={params['seed']}")

        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return

        original = tensors[key]
        w = original.float()
        d_out, d_in = w.shape

        print(f"  Target: {key} (shape {d_out}x{d_in})")

        U = torch.randn(d_out, rank)
        V = torch.randn(rank, d_in)
        N = (U @ V) / math.sqrt(d_in)

        w_bent = w + alpha * N

        noise_norm = (alpha * N).norm().item()
        change_norm = (w_bent - w).norm().item()
        weight_norm = w.norm().item()

        tensors[key] = w_bent.to(original.dtype)
        self.check_bf16_survival(key, original, w_bent)
        print(f"  {key}: rank={rank}, alpha={alpha}, "
              f"noise_norm={noise_norm:.6f}, change_norm={change_norm:.6f}, "
              f"weight_norm={weight_norm:.4f}, noise/weight={noise_norm/weight_norm:.6e}")
        print(f"  Modified 1 tensor(s)")

    def _apply_latent_channel(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        """Apply near-orthogonal rotation to latent input projection."""
        alpha = params["alpha"]

        key = "x_embedder.weight"

        print(f"  embed_warp(latent_channel): alpha={alpha}, seed={params['seed']}")

        if key not in tensors:
            print(f"  WARNING: key '{key}' not found in tensors, skipping")
            return

        original = tensors[key]
        w = original.float()
        d_out, d_in = w.shape

        print(f"  Target: {key} (shape {d_out}x{d_in})")

        # Generate orthogonal matrix Q via QR decomposition of random matrix
        # Q will be (d_out, d_out) — a rotation in the output (hidden) space
        random_matrix = torch.randn(d_out, d_out)
        Q, R = torch.linalg.qr(random_matrix)
        # Fix sign ambiguity: ensure positive diagonal in R
        # This makes Q deterministic given the same random_matrix
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.unsqueeze(0)

        # Near-orthogonal interpolation: M = (1 - alpha) * I + alpha * Q
        I = torch.eye(d_out)
        M = (1.0 - alpha) * I + alpha * Q

        # Apply: W' = M @ W
        w_bent = M @ w

        change_norm = (w_bent - w).norm().item()
        weight_norm = w.norm().item()

        # Log rotation properties
        m_det = torch.linalg.det(M).item()
        m_cond = torch.linalg.cond(M).item()

        tensors[key] = w_bent.to(original.dtype)
        self.check_bf16_survival(key, original, w_bent)
        print(f"  {key}: alpha={alpha}, "
              f"change_norm={change_norm:.4f}, weight_norm={weight_norm:.4f}, "
              f"change/weight={change_norm/weight_norm:.4e}")
        print(f"  Rotation matrix M: det={m_det:.6f}, cond={m_cond:.4f}")
        print(f"  Modified 1 tensor(s)")
