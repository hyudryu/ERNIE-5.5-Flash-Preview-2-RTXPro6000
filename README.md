# ERNIE-5.5-Flash-Preview on 2× RTX PRO 6000 Blackwell

Recipe, SM120 patch and benchmarks for serving
[baidu/ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)
(280B-parameter MoE, 14B active, 1M context) with vLLM on two 96 GB RTX PRO 6000
Blackwell workstation cards (SM120), managed through SparkDeck.

Every number below is **TP2, FP8 KV cache, MXFP4 experts / FP8 dense as shipped
(no extra quantization)**, DSpark speculative decoding, thinking on (the model
default).

## Status

| | Baseline |
| --- | --- |
| vLLM | 0.32.0 + [vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) `ba322a5` |
| Patch | SM120 attention port (1 file) |
| Layout | TP2, FP8 KV |
| DSpark draft tokens | 7, probabilistic |
| Max sequences | 8 |
| Context per session | 140,000 |
| KV pool | 140,304 tokens (5.83 GiB per card) |

The stock plugin does not start on SM120. The baseline is the smallest patch
that makes Baidu's recommended flags run correctly on these cards, with
nothing tuned for speed.

## Baseline results

Quick runs: one pass per cell, 1,024 output tokens, temperature 0.
`ms/step` is the wall time of one decode step (draft + verify) and `tok/step`
is tokens emitted per step, both from `/metrics` deltas.

### Decode, no context (C1–C8)

| Workload | C | Aggregate tok/s | Per-user tok/s | ms/step | tok/step | Draft acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| prose | 1 | 102.9 | 102.9 | 21.8 | 2.25 | 18% |
| prose | 2 | 156.9 | 78.4 | 28.5 | 2.24 | 18% |
| prose | 4 | 216.4 | 54.1 | 42.4 | 2.29 | 18% |
| prose | 8 | 294.8 | 36.8 | 61.7 | 2.27 | 18% |
| coding | 1 | 163.9 | 163.9 | 23.0 | 3.77 | 40% |
| coding | 2 | 232.2 | 116.1 | 32.3 | 3.75 | 39% |
| coding | 4 | 296.5 | 74.1 | 50.3 | 3.73 | 39% |
| coding | 8 | 438.8 | 54.9 | 76.6 | 4.20 | 46% |
| counting | 1 | 228.3 | 228.3 | 22.6 | 5.17 | 60% |
| counting | 2 | 448.9 | 224.4 | 30.5 | 6.84 | 83% |
| counting | 4 | 655.2 | 163.8 | 43.2 | 7.07 | 87% |
| counting | 8 | 871.6 | 109.0 | 63.2 | 6.88 | 84% |

| Mean of the three workloads | C1 | C2 | C4 | C8 |
| --- | --- | --- | --- | --- |
| Aggregate tok/s | 165.0 | 279.3 | 389.4 | 535.1 |
| ms/step | 22.5 | 30.4 | 45.3 | 67.2 |
| tok/step | 3.73 | 4.28 | 4.36 | 4.45 |

### Prefill

Unique documents, so nothing is served from the prefix cache. Two warm rounds.

| Prompt tokens | TTFT (s) | Prefill tok/s |
| --- | --- | --- |
| 1K | 0.20 | 5,130 |
| 4K | 0.60 | 6,860 |
| 16K | 2.31 | 7,120 |
| 32K | 4.60–5.44 | 6,020–7,130 |
| 64K | 9.44 | 6,920 |
| 128K | 20.5 | 6,370 |

### Decode with context

Prose after a filler document of the given size, 512 output tokens. Cells that
do not fit the 140,304-token pool are left out.

| Context | C1 per-user tok/s | C2 aggregate | C4 aggregate | TTFT at C1 (s) |
| --- | --- | --- | --- | --- |
| 0 | 102.9 | 156.9 | 216.4 | — |
| 16K | 108.2 | 167.4 | 216.9 | 2.3 |
| 32K | 109.9 | 160.7 | 208.6 | 4.6 |
| 64K | 110.2 | 162.3 | — | 9.7 |
| 128K | 105.7 | — | — | 21.5 |

Decode speed does not depend on context length. At 16K, C8 ran at 21.2
tok/s per user; the requests never all decoded at once, so there is no
aggregate figure for it.

## What the baseline needed on SM120

1. **Attention kernels.** The plugin runs both attention types on FlashMLA,
   which ships SM90/SM100 kernels only, and refuses to start
   (`ERNIE 5.5 DSA needs the sink-capable FLASHMLA_SPARSE backend: compute
   capability not supported`). `patch/sm120` routes them to FlashInfer's SM120
   sparse-MLA kernels instead, keeping the same packed `fp8_ds_mla` KV layout:
   - HCA layers (37 of 43) use vLLM's `DeepseekV4FlashInferSM120Attention`.
   - DSA layers (6 of 43) use vLLM's `FLASHINFER_MLA_SPARSE_SM120` backend with
     the per-head attention sink passed through.
2. **Padded rows under CUDA graphs.** The SM120 HCA kernels write only the rows
   of real tokens. With CUDA graphs the batch is padded (25 tokens run as 32),
   and the untouched rows grew to inf/NaN over a few layers and then faulted in
   the expert routing (`CUDA error: an illegal memory access`). The patch zeroes
   them.
3. **`--disable-custom-all-reduce`.** vLLM's custom all-reduce faults on this
   machine's peer-to-peer setup.
4. **Memory.** Weights take 80.45 GiB of each 94.97 GiB card at TP2, which
   leaves about 12.8 GiB for activations, CUDA graphs and KV cache:
   - 64 sequences (first try): graphs take 5.07 GiB and only a 7.6K context fits.
   - 8 sequences at `--gpu-memory-utilization 0.985`: 6.85 GiB of KV and a 215K
     context, but a 64K prefill then runs out of memory, because real prefill
     needs more than the 4.49 GiB vLLM profiles at startup.
   - 8 sequences with `--kv-cache-memory 6264723129` (5.83 GiB): 140K context,
     stable through a 128K prefill. Peak use was 97,009 of 97,887 MiB.

   Reaching the 500K target needs about 10.6 GiB of KV per card, so it is a
   memory problem for the revisions, not a flag.

## Setup

```bash
# 1. Image: vLLM 0.32.0 + the plugin, pinned
docker build -t vllm/vllm-openai:ernie-5.5-flash recipe/

# 2. Weights (166 GB), pinned revision
hf download baidu/ERNIE-5.5-Flash-Preview --revision ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe

# 3. SparkDeck deployment (replace <repo> in the mount path first)
curl -X POST http://localhost:7878/api/v1/deployments \
  -H 'Origin: http://localhost:7878' -H 'Content-Type: application/json' \
  -d @recipe/sparkdeck-baseline-tp2.json
```

The deployment mounts one file over the plugin inside the container:

```
patch/sm120/vllm_ernie/models/ernie5_5/nvidia/attention.py
  -> /opt/vLLM-ERNIE/src/vllm_ernie/models/ernie5_5/nvidia/attention.py
```

Flags (`recipe/sparkdeck-baseline-tp2.json`):

```
vllm serve baidu/ERNIE-5.5-Flash-Preview --revision ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe
  --trust-remote-code -tp 2 --disable-custom-all-reduce
  --kv-cache-dtype fp8 --kv-cache-memory 6264723129
  --max-model-len 140000 --max-num-seqs 8
  --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Environment: `HF_HUB_OFFLINE=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`. Startup
takes about 4.5 minutes (1 minute of weights once cached, the rest is graph
capture and kernel warm-up).

Differences from Baidu's recipe: TP2 instead of DP+EP (the experts are split
across the two cards either way, and TP2 is what fits), no MXFP4 indexer cache
(data-centre Blackwell only, so the indexer keeps FP8), and the memory flags
above.

## Benchmarks

```bash
cd bench
python3 step_matrix.py <label> 1,2,4,8          # decode: tok/s, ms/step, tok/step
python3 bench.py --only prefill --out p.json    # prefill sweep
python3 bench.py --only context --pool-tokens <pool> --out c.json
```

Raw results are in `bench/results/`, the startup log in `docs/logs/`.
