"""Model architecture info for FLUX.2 Klein Base 4B, populated from Prompt 0 audit."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    """Immutable container for model architecture metadata."""

    num_double_blocks: int
    num_single_blocks: int
    hidden_size: int
    num_heads: int
    head_dim: int
    mlp_hidden_dim: int
    mlp_ratio: float
    in_channels: int
    joint_attention_dim: int
    patch_size: int
    rope_theta: int
    axes_dims_rope: tuple[int, ...]

    double_block_key_templates: tuple[str, ...] = field(repr=False)
    single_block_key_templates: tuple[str, ...] = field(repr=False)
    modulation_keys: tuple[str, ...] = field(repr=False)
    embedder_keys: tuple[str, ...] = field(repr=False)
    final_layer_keys: tuple[str, ...] = field(repr=False)

    key_mapping: dict[str, str] = field(default_factory=dict, repr=False)

    def double_block_keys(self, block_idx: int) -> list[str]:
        """Return all tensor keys for a specific double block."""
        return [t.format(N=block_idx) for t in self.double_block_key_templates]

    def single_block_keys(self, block_idx: int) -> list[str]:
        """Return all tensor keys for a specific single block."""
        return [t.format(N=block_idx) for t in self.single_block_key_templates]

    def all_double_block_keys(self) -> list[str]:
        """Return all tensor keys across all double blocks, sorted."""
        keys = []
        for i in range(self.num_double_blocks):
            keys.extend(self.double_block_keys(i))
        return sorted(keys)

    def all_single_block_keys(self) -> list[str]:
        """Return all tensor keys across all single blocks, sorted."""
        keys = []
        for i in range(self.num_single_blocks):
            keys.extend(self.single_block_keys(i))
        return sorted(keys)

    @classmethod
    def klein_4b(cls) -> ModelInfo:
        """Factory for FLUX.2 Klein Base 4B architecture (from Prompt 0 audit)."""

        double_block_key_templates = (
            "transformer_blocks.{N}.attn.add_k_proj.weight",
            "transformer_blocks.{N}.attn.add_q_proj.weight",
            "transformer_blocks.{N}.attn.add_v_proj.weight",
            "transformer_blocks.{N}.attn.norm_added_k.weight",
            "transformer_blocks.{N}.attn.norm_added_q.weight",
            "transformer_blocks.{N}.attn.norm_k.weight",
            "transformer_blocks.{N}.attn.norm_q.weight",
            "transformer_blocks.{N}.attn.to_add_out.weight",
            "transformer_blocks.{N}.attn.to_k.weight",
            "transformer_blocks.{N}.attn.to_out.0.weight",
            "transformer_blocks.{N}.attn.to_q.weight",
            "transformer_blocks.{N}.attn.to_v.weight",
            "transformer_blocks.{N}.ff.linear_in.weight",
            "transformer_blocks.{N}.ff.linear_out.weight",
            "transformer_blocks.{N}.ff_context.linear_in.weight",
            "transformer_blocks.{N}.ff_context.linear_out.weight",
        )

        single_block_key_templates = (
            "single_transformer_blocks.{N}.attn.norm_k.weight",
            "single_transformer_blocks.{N}.attn.norm_q.weight",
            "single_transformer_blocks.{N}.attn.to_out.weight",
            "single_transformer_blocks.{N}.attn.to_qkv_mlp_proj.weight",
        )

        modulation_keys = (
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
        )

        embedder_keys = (
            "context_embedder.weight",
            "x_embedder.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.timestep_embedder.linear_2.weight",
        )

        final_layer_keys = (
            "norm_out.linear.weight",
            "proj_out.weight",
        )

        # Klein 4B: raw safetensors keys = diffusers keys (identity mapping)
        key_mapping: dict[str, str] = {}

        return cls(
            num_double_blocks=5,
            num_single_blocks=20,
            hidden_size=3072,
            num_heads=24,
            head_dim=128,
            mlp_hidden_dim=9216,
            mlp_ratio=3.0,
            in_channels=128,
            joint_attention_dim=7680,
            patch_size=1,
            rope_theta=2000,
            axes_dims_rope=(32, 32, 32, 32),
            double_block_key_templates=double_block_key_templates,
            single_block_key_templates=single_block_key_templates,
            modulation_keys=modulation_keys,
            embedder_keys=embedder_keys,
            final_layer_keys=final_layer_keys,
            key_mapping=key_mapping,
        )
