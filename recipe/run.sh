#!/usr/bin/env bash
# Serve ERNIE-5.5-Flash-Preview on 2x RTX PRO 6000 Blackwell (TP2, FP8 KV cache).
# usage: recipe/run.sh [baseline|v1|v2]      (default v2)
#   DRY_RUN=1 prints the docker command instead of running it.
#   PORT, HF_CACHE and NAME can be overridden from the environment.
set -euo pipefail
VERSION=${1:-v2}
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=vllm/vllm-openai:ernie-5.5-flash
PLUGIN=/opt/vLLM-ERNIE/src
SITE=/usr/local/lib/python3.12/dist-packages
ATT=vllm_ernie/models/ernie5_5/nvidia/attention.py
MOE=vllm_ernie/models/ernie5_5/nvidia/moe.py
GEMM=vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
KVU=vllm/v1/core/kv_cache_utils.py

DSPARK='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
ADAPTIVE='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
MOUNTS=(); ENVS=(); FLAGS=()
case "$VERSION" in
  baseline)
    MOUNTS=(-v "$REPO/patch/sm120/$ATT:$PLUGIN/$ATT:ro")
    FLAGS=(--max-model-len 140000 --kv-cache-memory 6264723129 --speculative-config "$DSPARK") ;;
  v1)
    P=$REPO/patch/v1
    MOUNTS=(-v "$P/$ATT:$PLUGIN/$ATT:ro" -v "$P/$MOE:$PLUGIN/$MOE:ro" -v "$P/$GEMM:$SITE/$GEMM:ro")
    FLAGS=(--max-model-len 524288 --kv-cache-memory 8391956153 --max-num-batched-tokens 2048
           --speculative-config "$ADAPTIVE") ;;
  v2)
    P=$REPO/patch/v2
    MOUNTS=(-v "$P/$ATT:$PLUGIN/$ATT:ro" -v "$P/$MOE:$PLUGIN/$MOE:ro" -v "$P/$GEMM:$SITE/$GEMM:ro"
            -v "$P/$KVU:$SITE/$KVU:ro")
    ENVS=(-e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1)
    FLAGS=(--max-model-len 300000 --kv-cache-memory 9680446000 --max-num-batched-tokens 1024
           --speculative-config "$ADAPTIVE") ;;
  *) echo "unknown version: $VERSION" >&2; exit 2 ;;
esac

CMD=(docker run -d --name "${NAME:-ernie-5.5-flash}" --gpus all --ipc=host
  -p "${PORT:-8000}:8000"
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 "${ENVS[@]}" "${MOUNTS[@]}"
  "$IMAGE" baidu/ERNIE-5.5-Flash-Preview
  --revision ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe --served-model-name ernie-5.5-flash
  --trust-remote-code -tp 2 --disable-custom-all-reduce
  --kv-cache-dtype fp8 --max-num-seqs 8
  --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice
  --enable-prompt-tokens-details "${FLAGS[@]}")
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
