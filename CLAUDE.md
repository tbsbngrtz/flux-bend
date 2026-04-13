# CLAUDE.md — flux-bend

## Project

`flux-bend` is a deterministic model bending / glitching tool for FLUX.2 Klein Base 4B.
It modifies diffusion transformer weights at the safetensors level before inference
to produce glitched outputs for artistic research and a Master's thesis on weight-space
interventions in diffusion image transformers.

## Model

Target: FLUX.2 Klein Base 4B (bf16, Apache 2.0).
HuggingFace ID: `black-forest-labs/FLUX.2-klein-base-4B`
No other models. No FP8. No distilled variants. Base only.

Model access: always via huggingface_hub. The tool accepts an HF model ID
(e.g. `black-forest-labs/FLUX.2-klein-base-4B`) or a local path.
When given an HF ID:
- `huggingface_hub.snapshot_download(model_id)` downloads/caches and returns the local path.
- Transformer safetensors are in the `transformer/` subfolder (may be sharded).
- For inference, `Flux2KleinPipeline.from_pretrained(model_id)` uses the same cache.
- NEVER modify files in the HF cache. Bent tensors go to the outputs directory.

HF auth: `huggingface_hub.login()` or `HF_TOKEN` env var. The tool does not store tokens.

Architecture (verified by Prompt 0 audit):
- Klein 4B Base: 5 double-stream + 20 single-stream blocks
- hidden_size = 3072, num_heads = 24, head_dim = 128
- mlp_hidden_dim = 9216 (ratio 3.0x), SwiGLU activation
- Shared AdaLN-Zero modulation (one for double blocks img, one for txt, one for single blocks — NOT per-block)
- RoPE positional embeddings (computed at runtime, rope_theta=2000)
- 128 VAE latent channels (post pixel-shuffle), patch_size = 1
- joint_attention_dim = 7680 (Qwen3-4B text encoder, bundled)
- guidance_embeds = False (no separate guidance embedder)
- All weights bf16, no biases anywhere
- Inference: diffusers Flux2KleinPipeline, base model uses ~50 steps, guidance_scale=4.0
- 3,875,544,576 parameters (3.88B), 169 tensors

### Tensor key patterns (double-stream)

5 double-stream blocks (indices 0–4), 16 weight tensors per block.
All weights are bf16, no biases.

| Sub-key | Shape | Description |
|---|---|---|
| `transformer_blocks.N.attn.add_k_proj.weight` | (3072, 3072) | Text stream key projection |
| `transformer_blocks.N.attn.add_q_proj.weight` | (3072, 3072) | Text stream query projection |
| `transformer_blocks.N.attn.add_v_proj.weight` | (3072, 3072) | Text stream value projection |
| `transformer_blocks.N.attn.norm_added_k.weight` | (128,) | RMSNorm on text keys |
| `transformer_blocks.N.attn.norm_added_q.weight` | (128,) | RMSNorm on text queries |
| `transformer_blocks.N.attn.norm_k.weight` | (128,) | RMSNorm on image keys |
| `transformer_blocks.N.attn.norm_q.weight` | (128,) | RMSNorm on image queries |
| `transformer_blocks.N.attn.to_add_out.weight` | (3072, 3072) | Text stream attention output |
| `transformer_blocks.N.attn.to_k.weight` | (3072, 3072) | Image stream key projection |
| `transformer_blocks.N.attn.to_out.0.weight` | (3072, 3072) | Image stream attention output |
| `transformer_blocks.N.attn.to_q.weight` | (3072, 3072) | Image stream query projection |
| `transformer_blocks.N.attn.to_v.weight` | (3072, 3072) | Image stream value projection |
| `transformer_blocks.N.ff.linear_in.weight` | (18432, 3072) | Image MLP up-proj (SwiGLU, out = 2 × 9216) |
| `transformer_blocks.N.ff.linear_out.weight` | (3072, 9216) | Image MLP down-proj (9216 → 3072) |
| `transformer_blocks.N.ff_context.linear_in.weight` | (18432, 3072) | Text MLP up-proj (SwiGLU, out = 2 × 9216) |
| `transformer_blocks.N.ff_context.linear_out.weight` | (3072, 9216) | Text MLP down-proj (9216 → 3072) |

Note: Unlike the 9B model, double-block attention is NOT fused — Q/K/V are separate projections.
Text stream uses `add_q_proj`/`add_k_proj`/`add_v_proj` (not `to_add_q`/`to_add_k`/`to_add_v`).

### Tensor key patterns (single-stream)

20 single-stream blocks (indices 0–19), 4 weight tensors per block.

| Sub-key | Shape | Description |
|---|---|---|
| `single_transformer_blocks.N.attn.norm_k.weight` | (128,) | RMSNorm on keys |
| `single_transformer_blocks.N.attn.norm_q.weight` | (128,) | RMSNorm on queries |
| `single_transformer_blocks.N.attn.to_out.weight` | (3072, 12288) | Fused attn+MLP output (3072 attn + 9216 MLP = 12288) |
| `single_transformer_blocks.N.attn.to_qkv_mlp_proj.weight` | (27648, 3072) | Fused QKV+MLP input (9216 QKV + 18432 MLP = 27648) |

Fused projection layout:
- `to_qkv_mlp_proj` output dim 27648: first 9216 = QKV (3072 Q + 3072 K + 3072 V), remaining 18432 = MLP input (SwiGLU gated, 2 × 9216)
- `to_out` input dim 12288: first 3072 = attention output, remaining 9216 = MLP output

### Non-block tensors (embedders, final layer, modulation)

| Key | Shape | Description |
|---|---|---|
| `context_embedder.weight` | (3072, 7680) | Text encoder output → hidden dim |
| `double_stream_modulation_img.linear.weight` | (18432, 3072) | Shared img modulation for all 5 double blocks (6 groups × 3072) |
| `double_stream_modulation_txt.linear.weight` | (18432, 3072) | Shared txt modulation for all 5 double blocks (6 groups × 3072) |
| `norm_out.linear.weight` | (6144, 3072) | Final AdaLN modulation (2 groups × 3072: shift + scale) |
| `proj_out.weight` | (128, 3072) | Final projection to latent patch space |
| `single_stream_modulation.linear.weight` | (9216, 3072) | Shared modulation for all 20 single blocks (3 groups × 3072) |
| `time_guidance_embed.timestep_embedder.linear_1.weight` | (3072, 256) | Timestep embedder layer 1 |
| `time_guidance_embed.timestep_embedder.linear_2.weight` | (3072, 3072) | Timestep embedder layer 2 |
| `x_embedder.weight` | (3072, 128) | Latent input projection (128 → 3072) |

Shared AdaLN modulation groups:
- **double_stream_modulation_img/txt**: 6 groups of 3072 = shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff (shared across ALL 5 double blocks)
- **single_stream_modulation**: 3 groups of 3072 = shift, scale, gate (shared across ALL 20 single blocks)
- **norm_out**: 2 groups of 3072 = shift, scale (final layer normalization)

RoPE: frequencies computed at runtime (not stored in weights). `rope_theta=2000`, `axes_dims_rope=[32, 32, 32, 32]`.

### Diffusers key mapping

The Klein 4B safetensors file uses **diffusers-format keys directly**.
Raw key = diffusers key (identity mapping). No remapping needed.

This is different from the 9B model which uses BFL-format keys like `double_blocks.N.img_mlp.0.weight`
that diffusers remaps to `transformer_blocks.N.ff.linear_in.weight` during loading.

Since `flux-bend` operates at the safetensors level, we can use diffusers attribute paths directly
as tensor keys — no translation layer needed.

### Transformer safetensors file structure

Single file (not sharded): `transformer/diffusion_pytorch_model.safetensors` (7392.0 MB)

169 tensors, 3,875,544,576 parameters (3.88B), all bf16, no biases.

## Determinism — ABSOLUTE REQUIREMENT

Every output must be perfectly reproducible from its parameter set alone.

- NO `torch.randperm`, NO `torch.rand` for selection. Head indices are explicit lists.
- `seed` is MANDATORY (no default) for any operation that involves generated matrices
  (low-rank noise, orthogonal rotations, SVD).
- Inference `seed` is MANDATORY in every config.
- `torch.manual_seed(seed)` is called exactly once at the start of `apply()`.
  The iteration order over tensor keys is sorted alphabetically and documented.
- No optional randomness. No `frac_heads`. No random permutations.
- The sweep manifest + config must be sufficient to reproduce any output exactly.

## Hardware

- Machine: wednesday (Windows, RTX 5070 Ti, 16 GB VRAM)
- Memory strategy: clone + bend on CPU, patch pipeline state_dict, use model_cpu_offload
- System RAM needed: ~20 GB (8 GB base model + 8 GB clone + overhead)

## Tech stack

- Python 3.12, managed with uv
- PyTorch (CUDA 12.8), safetensors, diffusers, huggingface_hub, transformers,
  Pillow, pyyaml, alive-progress
- Optional [analysis]: lpips, torchmetrics
- All paths via pathlib, no hardcoded separators

## Environment setup

The project uses uv for environment and dependency management.
PyTorch with CUDA requires the `--extra-index-url` for the cu128 wheel index.

Setup: `setup.bat` (Windows) or `setup.sh` (Linux/macOS)
Run:   `run.bat <subcommand> [args]` or `run.sh <subcommand> [args]`

These scripts are thin wrappers — they ensure the venv exists and call
`uv run flux-bend <args>`. All CLI args are passed through.

Do NOT install packages manually with pip. Use uv only.
Do NOT activate the venv manually. Use the run scripts or `uv run`.

## Code conventions

- Type hints on every function signature.
- pathlib.Path for all file paths, never raw strings.
- All weight math in float32, cast back to original dtype.
- After casting back to bf16, check if the intervention survived the precision floor:
  compute `(bent_bf16.float() - original_bf16.float()).abs().max()`.
  Log a warning if it's zero (intervention below bf16 precision).
- Each bending mode is one file in `src/flux_bend/modes/`, one class per file.
- Mode class `name` attribute must match the filename (without `.py`).
- Logging: print which tensor keys were modified, head indices, norms of changes.
- No matplotlib dependency. Charts via Pillow drawing only.

## File structure

```
flux-bend/
├── CLAUDE.md
├── pyproject.toml
├── setup.bat               # Windows: create venv + install deps
├── setup.sh                # Linux/macOS: create venv + install deps
├── run.bat                 # Windows: uv run flux-bend <args>
├── run.sh                  # Linux/macOS: uv run flux-bend <args>
├── .python-version         # pins Python 3.12
├── .gitignore
├── configs/
│   └── *.yaml
├── src/
│   └── flux_bend/
│       ├── __init__.py
│       ├── registry.py
│       ├── model_info.py
│       ├── loader.py
│       ├── sweep.py
│       ├── naming.py
│       ├── inference.py
│       ├── grid.py
│       ├── cli.py
│       └── modes/
│           ├── __init__.py
│           ├── _base.py
│           ├── attn_head_scale.py
│           ├── mlp_low_rank.py
│           ├── adaln_permute.py
│           ├── svd_boost.py
│           ├── embed_warp.py
│           ├── cross_stream.py
│           ├── block_dropout.py
│           ├── quant_noise.py
│           ├── rope_warp.py
│           └── final_layer_warp.py
└── outputs/              # gitignored
```

## Sweep system

- YAML configs define sweeps: lists expand into cartesian products.
- Grid layout: `grid.rows` and `grid.cols` specify which param maps to which axis.
- Maximum 2 swept params per grid. 3+ swept params → one grid per value of the 3rd param.
- Baseline (unbent) image is always job 0.
- Progress: alive-progress bar over jobs, per-job timing logged after completion.
- `--dry-run` prints all jobs and grid layout without executing.

## What this project is NOT

- Not a general-purpose model editor. Klein 4B only.
- Not a training tool. Inference-time weight manipulation only.
- Not a LoRA/fine-tuning system. Direct tensor modification.
- Not a ComfyUI node (yet).
