# FLUX.2 Klein Base 4B — Architecture Audit Summary

## 1. Total Tensor & Parameter Count

- **Raw safetensors keys:** 169
- **Diffusers state_dict keys:** 169
- **Total parameters:** 3,875,544,576 (3.88B)
- **Key format:** Raw safetensors uses diffusers-format keys directly (identity mapping)

## 2. Transformer File Structure

Single file (not sharded): `diffusion_pytorch_model.safetensors` (7392.0 MB)

## 3. Unique Key Prefixes

| Prefix | Count | Description |
|---|---|---|
| `context_embedder` | 1 | Text encoder output projection |
| `double_stream_modulation_img` | 1 | Shared AdaLN modulation (double blocks, image stream) |
| `double_stream_modulation_txt` | 1 | Shared AdaLN modulation (double blocks, text stream) |
| `norm_out` | 1 | Final output normalization (AdaLN) |
| `proj_out` | 1 | Final output projection to latent space |
| `single_stream_modulation` | 1 | Shared AdaLN modulation (single blocks) |
| `single_transformer_blocks.N` | 80 | Single-stream blocks (20 blocks × 4 keys) |
| `time_guidance_embed` | 2 | Timestep embedding |
| `transformer_blocks.N` | 80 | Double-stream blocks (5 blocks × 16 keys) |
| `x_embedder` | 1 | Latent input projection |

## 4. Double-Stream Block Sub-Key Patterns

**5 double-stream blocks** (indices 0–4), 16 keys per block

| Sub-key | Shape | Description |
|---|---|---|
| `attn.add_k_proj.weight` | (3072, 3072) | Text stream key projection |
| `attn.add_q_proj.weight` | (3072, 3072) | Text stream query projection |
| `attn.add_v_proj.weight` | (3072, 3072) | Text stream value projection |
| `attn.norm_added_k.weight` | (128,) | RMSNorm on text keys |
| `attn.norm_added_q.weight` | (128,) | RMSNorm on text queries |
| `attn.norm_k.weight` | (128,) | RMSNorm on image keys |
| `attn.norm_q.weight` | (128,) | RMSNorm on image queries |
| `attn.to_add_out.weight` | (3072, 3072) | Text stream attention output |
| `attn.to_k.weight` | (3072, 3072) | Image stream key projection |
| `attn.to_out.0.weight` | (3072, 3072) | Image stream attention output |
| `attn.to_q.weight` | (3072, 3072) | Image stream query projection |
| `attn.to_v.weight` | (3072, 3072) | Image stream value projection |
| `ff.linear_in.weight` | (18432, 3072) | Image MLP up-proj (SwiGLU, out = 2 × 9216) |
| `ff.linear_out.weight` | (3072, 9216) | Image MLP down-proj (9216 → 3072) |
| `ff_context.linear_in.weight` | (18432, 3072) | Text MLP up-proj (SwiGLU, out = 2 × 9216) |
| `ff_context.linear_out.weight` | (3072, 9216) | Text MLP down-proj (9216 → 3072) |

## 5. Single-Stream Block Sub-Key Patterns

**20 single-stream blocks** (indices 0–19), 4 keys per block

| Sub-key | Shape | Description |
|---|---|---|
| `attn.norm_k.weight` | (128,) | RMSNorm on keys |
| `attn.norm_q.weight` | (128,) | RMSNorm on queries |
| `attn.to_out.weight` | (3072, 12288) | Fused attn+MLP output (3072 attn + 9216 MLP = 12288) |
| `attn.to_qkv_mlp_proj.weight` | (27648, 3072) | Fused QKV+MLP input (9216 QKV + 18432 MLP = 27648) |

**Fused projection layout:**
- `to_qkv_mlp_proj` output dim 27648: first 9216 = QKV (3072 each), remaining 18432 = MLP input (SwiGLU gated)
- `to_out` input dim 12288: first 3072 = attention output, remaining 9216 = MLP output

## 6. Non-Block Tensors

| Key | Shape | Description |
|---|---|---|
| `context_embedder.weight` | (3072, 7680) | Text encoder output → hidden dim ((3072, 7680)) |
| `double_stream_modulation_img.linear.weight` | (18432, 3072) | Shared img modulation for all 5 double blocks |
| `double_stream_modulation_txt.linear.weight` | (18432, 3072) | Shared txt modulation for all 5 double blocks |
| `norm_out.linear.weight` | (6144, 3072) | Final AdaLN modulation |
| `proj_out.weight` | (128, 3072) | Final projection to latent patch space |
| `single_stream_modulation.linear.weight` | (9216, 3072) | Shared modulation for all 20 single blocks |
| `time_guidance_embed.timestep_embedder.linear_1.weight` | (3072, 256) | Timestep embedder |
| `time_guidance_embed.timestep_embedder.linear_2.weight` | (3072, 3072) | Timestep embedder |
| `x_embedder.weight` | (3072, 128) | Latent input projection (128 → 3072) |

## 7. Verified Dimensions

| Dimension | Value | Source |
|---|---|---|
| hidden_size | 3072 | config: 24 heads × 128 head_dim |
| hidden_size (verified) | 3072 | x_embedder.weight shape[0] |
| num_attention_heads | 24 | config |
| attention_head_dim | 128 | config |
| num_double_blocks | 5 | counted from keys |
| num_single_blocks | 20 | counted from keys |
| in_channels (latent) | 128 | x_embedder.weight shape[1] |
| mlp_hidden_dim | 9216 | ff.linear_out.weight shape[1] |
| mlp_ratio | 3.0 | mlp_hidden / hidden_size |
| joint_attention_dim | 7680 | config (text encoder dim) |
| guidance_embeds | False | config |
| rope_theta | 2000 | config |
| axes_dims_rope | [32, 32, 32, 32] | config |
| patch_size | 1 | config |

## 8. Shared AdaLN-Zero Modulation

Klein 4B uses **shared modulation** — one linear layer produces modulation signals for ALL blocks of each type.

### double_stream_modulation_img
- Shape: (18432, 3072)
- Output groups: 6 × 3072 = 18432
- Interpretation: 6 groups = shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff
- These 6 modulation vectors are applied to ALL 5 double blocks identically

### double_stream_modulation_txt
- Shape: (18432, 3072)
- Output groups: 6 × 3072 = 18432
- Same 6-group structure as img modulation, for text stream

### single_stream_modulation
- Shape: (9216, 3072)
- Output groups: 3 × 3072 = 9216
- Interpretation: 3 groups = shift, scale, gate (for fused attn+MLP)

### norm_out (final layer modulation)
- Shape: (6144, 3072)
- Output groups: 2 × 3072
- Interpretation: shift, scale for final AdaLN normalization

## 9. Timestep Embedder

| Key | Shape |
|---|---|
| `time_guidance_embed.timestep_embedder.linear_1.weight` | (3072, 256) |
| `time_guidance_embed.timestep_embedder.linear_2.weight` | (3072, 3072) |

Note: `guidance_embeds = False` — no separate guidance embedder. Timestep only.
Timestep embedding channels: 256

## 10. Image Input Embedder (x_embedder)

- Key: `x_embedder.weight`
- Shape: (3072, 128)
- Maps 128 latent channels → 3072 hidden dim
- No bias (weight-only)

## 11. Final Output Projection

- `proj_out.weight`: (128, 3072)
- `norm_out.linear.weight`: (6144, 3072)
- Maps 3072 hidden dim → 128 (= patch_size² × out_channels = 1 × 128)

## 12. RoPE (Rotary Position Embeddings)

**No RoPE frequency tensors in weights.** Frequencies are computed at runtime.
- `rope_theta`: 2000
- `axes_dims_rope`: [32, 32, 32, 32]

No RoPE keys in diffusers state_dict either — confirmed runtime computation.

## Key Mapping Note

The raw safetensors file for Klein 4B uses **diffusers-format keys directly**.
This means the raw-to-diffusers key mapping is an **identity mapping** — every key in the
safetensors file has the exact same name in `pipe.transformer.state_dict()`.

This is different from the 9B model, which uses a BFL-format key naming convention
(e.g., `double_blocks.N.img_mlp.0.weight`) that diffusers remaps during loading.

## Differences from FLUX.2 Klein 9B

| Feature | 9B | 4B |
|---|---|---|
| Parameters | ~9B | 3.88B |
| Double blocks | 8 | 5 |
| Single blocks | 24 | 20 |
| Hidden size | 4096 | 3072 |
| Attention heads | 32 | 24 |
| Head dim | 128 | 128 |
| Safetensors key format | BFL-format (requires remapping) | Diffusers-format (identity) |
| Biases | Yes (in most layers) | No (weight-only) |
| Guidance embeds | True | False |
| Joint attention dim | 15360 | 7680 |
| Double block QKV | Fused (img_attn.qkv) | Split (to_q, to_k, to_v) |
| Per-block modulation | Yes (in each block) | No (shared globally) |

