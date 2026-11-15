# GroupMask

Official implementation of **GroupMask: Layer-Adaptive Group-wise Sparsity for Semi-Structured LLM Pruning**.

GroupMask prunes contiguous weight groups (for example, 1×64 row segments) under one global sparsity budget, so each linear projection can keep a different fraction of its weights. A shared hypernetwork generates binary group masks, trained with deterministic hard gates, a straight-through estimator, a sparsity-budget regularizer, and self-distillation, while the pretrained weights stay frozen. The exported model is an exact sub-network of the original. Its retained groups are stored in a compressed format and executed with Triton decoding kernels.

## Repository layout

| Path | Contents |
|---|---|
| `train_groupmask.py` | Mask training (hypernetwork, hard gates + STE, global budget, self-distillation) |
| `export_groupmask.py` | Applies the learned gates and saves a standard Hugging Face model whose pruned groups are exactly zero |
| `scripts/run_groupmask.sh` | End-to-end pipeline: train → export → perplexity → zero-shot |
| `scripts/eval_ppl.py` | WikiText-2 perplexity (non-overlapping and sliding windows) |
| `scripts/zsavg.py` | Six-task zero-shot average from `lm_eval` output |
| `scripts/owl_ratios.py` | OWL per-block keep ratios (allocation control) |
| `scripts/extract_masks.py` | Extracts group masks from an exported model for the kernels |
| `scripts/layer_sparsity.py` | Per-layer, per-projection sparsity of an exported model |
| `scripts/seeded_run.py` | Seeds torch / NumPy / Python before running a recipe |
| `flashlm/` | Model, data, distributed, and mask-generation modules |
| `kernels/gsparse.py` | Compressed group-sparse format and Triton decoding kernels |
| `kernels/sparse24.py` | 2:4 format and Triton decoding kernel used as a baseline |
| `kernels/bench_request.py` | Full-request benchmark (prefill + decoding) for dense, GroupMask, 2:4, and structured models |
| `kernels/bench_layers.py`, `kernels/bench_curve.py` | Per-layer correctness and latency-versus-batch benchmarks |
| `kernels/real_check.py` | Checks that the compressed model reproduces the exported masked model |
| `kernels/sweep_real.py`, `kernels/sweep24.py` | Kernel tuning on the target GPU |
| `kernels/tuned/` | Tuned kernel tables for the RTX PRO 6000 and B200 |

## Setup

Training and evaluation were run with Python 3.10, PyTorch 2.7, Transformers 4.55, and Datasets 4.0. The full environment is pinned in `requirements.txt`. Zero-shot evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
pip install -r requirements.txt
pip install lm_eval
```

The kernel benchmarks were run with PyTorch 2.9.1, Triton 3.5.1, and cuSPARSELt 0.7.1, which are needed on Blackwell GPUs.

Data and output locations are set through environment variables:

```bash
export FLASHLM_HF_CACHE_DIR=/path/to/hf_datasets_cache   # optional
export FLASHLM_DATA_ROOT=/path/to/local_datasets         # optional, for locally stored datasets
export GROUPMASK_OUTPUT_ROOT=./outputs                   # hypernetwork checkpoints
export C4_JSON=/path/to/c4-train.00000-of-01024.json.gz  # only for --dataset_list=['c4']
```

## Training, export, and evaluation

Main configuration (LLaMA-2-7B, 1×64 groups, 50% sparsity, WikiText-103 calibration, 40k steps, one GPU):

```bash
bash scripts/run_groupmask.sh llama2-7b-1x64 meta-llama/Llama-2-7b-hf 1 64 0.5 wiki
```

The arguments are `<tag> <model> <group rows> <group cols> <keep ratio> <dataset> [steps=40000] [hypernetwork heads=16]`. Other group shapes use the same script, for example `8 8` for 8×8 groups. Set `SEED=<int>` for an independent run.

The script writes the hypernetwork checkpoint under `$GROUPMASK_OUTPUT_ROOT`, the exported model under `exported/<tag>`, and the perplexity and zero-shot results to the log.

### Allocation controls

All controls keep the group pattern, the global budget, and the training budget fixed:

```bash
# Uniform: every projection keeps the same fraction
bash scripts/run_groupmask.sh l2-uniform meta-llama/Llama-2-7b-hf 1 64 0.5 wiki 40000 16 "--uniform_sparsity=true"

# OWL per-block ratios
python scripts/owl_ratios.py --model meta-llama/Llama-2-7b-hf --out owl.json
PER_BLOCK_KEEP_JSON=owl.json bash scripts/run_groupmask.sh l2-owl meta-llama/Llama-2-7b-hf 1 64 0.5 wiki 40000 16 "--uniform_sparsity=true"

# Learned per-block allocation (projections within a block are tied)
TIE_BLOCK_SIZE=7 bash scripts/run_groupmask.sh l2-perblock meta-llama/Llama-2-7b-hf 1 64 0.5 wiki
```

## Compressed inference

Extract the masks from an exported model, then run the full-request benchmark:

```bash
python scripts/extract_masks.py exported/llama2-7b-1x64 masks/llama2-7b-1x64.npz 1 64
cd kernels
export GSPARSE_TUNED=tuned/tuned_rows_rtx6000.json GSPARSE24_TUNED=tuned/tuned24_rtx6000.json
python bench_request.py --mode dense --batch 1 --prompt 128 --gens 128 --out request.jsonl
python bench_request.py --mode group --masks ../masks/llama2-7b-1x64.npz --batch 1 --prompt 128 --gens 128 --out request.jsonl
python bench_request.py --mode 2:4-best --batch 1 --prompt 128 --gens 128 --out request.jsonl
```

Use `--graph_prefill 1` to capture prefill in a CUDA graph as well, and `--real <exported dir>` to time with real weights. To tune the kernels for a different GPU, run `sweep_real.py` and `sweep24.py` and point `GSPARSE_TUNED` / `GSPARSE24_TUNED` at the resulting tables. `real_check.py` verifies that the compressed model reproduces the exported masked model.

## Citation

The citation will be added after the review period.

## License

MIT (see `LICENSE`).
