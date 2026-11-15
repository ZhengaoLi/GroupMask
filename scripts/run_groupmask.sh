#!/bin/bash
# Train GroupMask -> export the pruned model -> WikiText-2 perplexity -> zero-shot accuracy, on one GPU.
#
# Usage (run from the repository root):
#   bash scripts/run_groupmask.sh <TAG> <HF_MODEL> <GIN> <GOUT> <P_KEEP> <DATASET> [STEPS=40000] [HN_GROUPS=16] [EXTRA_TRAIN_FLAGS]
#
# Main configuration of the paper (LLaMA-2-7B, 1x64 groups, 50% sparsity, WikiText calibration):
#   bash scripts/run_groupmask.sh llama2-7b-1x64 meta-llama/Llama-2-7b-hf 1 64 0.5 wiki
#
# Allocation controls (same group pattern and budget):
#   uniform per-projection ratio:  EXTRA="--uniform_sparsity=true"
#   OWL per-block ratio:           PER_BLOCK_KEEP_JSON=owl.json with EXTRA="--uniform_sparsity=true"
#                                  (owl.json from: python scripts/owl_ratios.py --model <HF_MODEL> --out owl.json)
#   learned per-block allocation:  TIE_BLOCK_SIZE=7 (7 projections per LLaMA block)
#
# Environment variables:
#   SEED=<int>          seed torch/numpy/random and use dataset_seed=42+SEED (default: unseeded, dataset_seed=42)
#   GROUPMASK_OUTPUT_ROOT=<dir> where hypernetwork checkpoints are written (default: ./outputs)
#   C4_JSON=<file>      local C4 shard used when DATASET=c4
#   FLASHLM_HF_CACHE_DIR / FLASHLM_DATA_ROOT  dataset cache locations (see flashlm/data/huggingface_dataset.py)
set -uo pipefail
TAG=$1; MODEL=$2; GIN=$3; GOUT=$4; P=$5; DS=$6; STEPS=${7:-40000}; HNG=${8:-16}; EXTRA=${9:-}; SEED=${SEED:-}
if [ -n "$SEED" ]; then RUN_PY="scripts/seeded_run.py train_groupmask.py"; DSEED=$((42 + SEED)); else RUN_PY="train_groupmask.py"; DSEED=42; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
EXP=groupmask_${TAG}; OUT=${EXPORT_DIR:-exported}/${TAG}; mkdir -p logs
test ! -e "$OUT" || { echo "refusing to overwrite $OUT"; exit 1; }

torchrun --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) $RUN_PY \
  --exp_name="$EXP" --batch_size=1 --hf_model="$MODEL" \
  --compile_flag=false --use_fsdp=true --use_bf16=true --use_minipile=false \
  --dataset_seed="$DSEED" --dataset_list="['$DS']" --total_n_step="$STEPS" --save_interval=10000 \
  --groups_in_dim="$GIN" --groups_out_dim="$GOUT" --hn_groups="$HNG" --gamma=0.01 \
  --kd_loss=true --mix_loss=false --scale_weight=false --soft_rank=false \
  --use_reinmax=false --hard_flag=false --semi_params=false --use_ddp=false \
  --hn_block_size=2048 --simple_gate=false --p="$P" --T=0.4 --lam=16 \
  --share_qk=false --hidden_kd=false --adam_8bit=false $EXTRA \
  2>&1 | tee logs/train_${EXP}.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED|train"; exit 1; }
RUN=$(ls -td ${GROUPMASK_OUTPUT_ROOT:-outputs}/$EXP/*/ | head -1); HN=${RUN}hn-ckpt-final.pt
test -f "$HN" || { echo "FAILED|no final checkpoint in $RUN"; exit 1; }

# Export: apply the deterministic hard gates and save a standard Hugging Face model with pruned groups set to zero.
torchrun --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) export_groupmask.py \
  --hf_model "$MODEL" --dataset "wikitext" --semi_evaluate "true" --eval_baseline "false" \
  --hn_path "$HN" --hn_groups="$HNG" --block_size=1024 --batch_limit=2000 \
  --groups_in_dim="$GIN" --groups_out_dim="$GOUT" --simple_gate=false --gate_mode="hard" \
  --save_hf_dir "$OUT" --share_qk=false --is_simple_gate=false || { echo "FAILED|export"; exit 1; }

python scripts/eval_ppl.py --name "$TAG" --model "$OUT" --tokenizer "$MODEL" --slide 1024 --nonov 4096 || echo "FAILED|ppl"

lm_eval --model hf --model_args pretrained="$OUT",dtype=float32 \
  --tasks hellaswag,arc_easy,arc_challenge,boolq,piqa,winogrande --device cuda:0 --batch_size auto \
  2>&1 | tee logs/zeroshot_${TAG}.log || echo "FAILED|zeroshot"
python scripts/zsavg.py logs/zeroshot_${TAG}.log
