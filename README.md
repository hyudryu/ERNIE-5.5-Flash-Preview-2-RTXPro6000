# ERNIE-5.5-Flash-Preview on 2× RTX PRO 6000 Blackwell

Recipe, SM120 patch and benchmarks for serving
[baidu/ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)
(280B-parameter MoE, 14B active, 1M context) with vLLM on two 96 GB RTX PRO 6000
Blackwell workstation cards (SM120), managed through SparkDeck.

Every number below is **TP2, FP8 KV cache, MXFP4 experts / FP8 dense as shipped
(no extra quantization)**, DSpark speculative decoding, thinking on (the model
default).

## Status

| | Baseline | V1 |
| --- | --- | --- |
| vLLM | 0.32.0 + [vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) `ba322a5` | same |
| Patch | SM120 attention port (1 file) | 3 files (`patch/v1`) |
| Layout | TP2, FP8 KV | TP2, FP8 KV |
| DSpark draft tokens | 7, probabilistic | 7, probabilistic, adaptive verification |
| Max sequences | 8 | 8 |
| Prefill chunk | 8,192 | 2,048 |
| Context per session | 140,000 | **524,288** |
| KV pool | 140,304 tokens (5.83 GiB per card) | **543,803 tokens** (7.82 GiB per card) |
| Mean decode, C1 / C2 / C4 / C8 | 165 / 279 / 389 / 535 tok/s | **228 / 297 / 574 / 841 tok/s** |
| Mean step time, C1 / C2 / C4 / C8 | 22.5 / 30.4 / 45.3 / 67.2 ms | **19.5 / 25.5 / 30.6 / 40.8 ms** |

The stock plugin does not start on SM120. The baseline is the smallest patch
that makes Baidu's recommended flags run correctly on these cards, with
nothing tuned for speed. V1 adds exact kernel-path fixes, the memory layout
for a 524K context, and vLLM's adaptive verification. Nothing is quantized
beyond what the checkpoint ships.

## Results

Quick runs: one pass per cell, 1,024 output tokens, temperature 0.
`ms/step` is the wall time of one decode step (draft + verify) and `tok/step`
is tokens emitted per step, both from `/metrics` deltas. Run-to-run variation
is about 5%, mostly from draft acceptance (the drafter samples).

### Decode, no context (C1–C8)

| Workload | C | Baseline tok/s | V1 tok/s | Change | Baseline ms/step | V1 ms/step | Baseline tok/step | V1 tok/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prose | 1 | 102.9 | **125.6** | +22% | 21.8 | **17.7** | 2.25 | **2.23** |
| prose | 2 | 156.9 | **205.6** | +31% | 28.5 | **20.8** | 2.24 | **2.14** |
| prose | 4 | 216.4 | **337.9** | +56% | 42.4 | **25.8** | 2.29 | **2.18** |
| prose | 8 | 294.8 | **507.5** | +72% | 61.7 | **33.0** | 2.27 | **2.09** |
| coding | 1 | 163.9 | **189.6** | +16% | 23.0 | **20.4** | 3.77 | **3.87** |
| coding | 2 | 232.2 | **252.9** | +9% | 32.3 | **28.8** | 3.75 | **3.64** |
| coding | 4 | 296.5 | **429.9** | +45% | 50.3 | **34.9** | 3.73 | **3.75** |
| coding | 8 | 438.8 | **613.2** | +40% | 76.6 | **51.4** | 4.20 | **3.94** |
| counting | 1 | 228.3 | **370.0** | +62% | 22.6 | **20.4** | 5.17 | **7.55** |
| counting | 2 | 448.9 | **431.7** | -4% | 30.5 | **26.9** | 6.84 | **5.81** |
| counting | 4 | 655.2 | **954.8** | +46% | 43.2 | **31.2** | 7.07 | **7.44** |
| counting | 8 | 871.6 | **1401.2** | +61% | 63.2 | **38.0** | 6.88 | **6.66** |

Aggregate tok/s. The counting rows move with draft acceptance from run to run
(60% to 95% at C1), so read `ms/step` for the kernel-level change there.

| Mean of the three workloads | Baseline | Kernel fix only | V1 |
| --- | --- | --- | --- |
| C1 tok/s (ms/step) | 165.0 (22.5) | 226.8 (19.7) | **228.4 (19.5)** |
| C2 tok/s (ms/step) | 279.3 (30.4) | 312.7 (25.8) | **296.8 (25.5)** |
| C4 tok/s (ms/step) | 389.4 (45.3) | 550.1 (33.8) | **574.2 (30.6)** |
| C8 tok/s (ms/step) | 535.1 (67.2) | 788.4 (45.9) | **840.6 (40.8)** |

"Kernel fix only" is V1 without adaptive verification, on the baseline's
memory layout.

### Prefill

Unique documents, so nothing is served from the prefix cache. Warm rounds.

| Prompt tokens | Baseline TTFT (s) | V1 TTFT (s) | Baseline tok/s | V1 tok/s |
| --- | --- | --- | --- | --- |
| 1K | 0.20 | **0.18** | 5,130 | **5,770** |
| 4K | 0.60 | **0.59** | 6,860 | **6,970** |
| 16K | 2.31 | **2.30** | 7,120 | **7,090** |
| 32K | 4.60–5.44 | **4.74** | 6,020–7,130 | **6,930** |
| 64K | 9.44 | **9.79–10.5** | 6,920 | **6,190–6,680** |
| 128K | 20.5 | **20.2–23.4** | 6,370 | **5,590–6,460** |
| 262K | does not fit | **49.6** | — | **5,250** |
| 498K | does not fit | **118.9** | — | **4,190** |

Prefill speed is about the same up to 32K and up to 10% lower at 64K–128K:
V1 prefills in 2,048-token chunks to free the memory that makes the 524K
context possible. The fused activation kernel on its own makes prefill about
4% faster (7,500 / 7,140 / 6,640 tok/s at 16K / 64K / 128K with 8,192-token
chunks).

### Decode with context

Prose after a filler document of the given size, 512 output tokens. Aggregate
tok/s while all requests decode; "—" means the cell does not fit the KV pool.

| Context | Baseline C1 | V1 C1 | Baseline C2 | V1 C2 | Baseline C4 | V1 C4 | Baseline C8 | V1 C8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 102.9 | **125.6** | 156.9 | **205.6** | 216.4 | **337.9** | 294.8 | **507.5** |
| 16K | 108.2 | **125.6** | 167.4 | **208.1** | 216.9 | **337.3** | n/a | **493.1** |
| 32K | 109.9 | **125.0** | 160.7 | **198.9** | 208.6 | **308.7** | — | **442.1** |
| 64K | 110.2 | **131.5** | 162.3 | **191.1** | — | **295.3** | — | n/a |
| 128K | 105.7 | **123.1** | — | **181.4** | — | **316.2** | — | — |

"n/a": the requests did not all decode at the same time, so there is no
steady aggregate. At C1 decode speed does not depend on context length. At
higher concurrency the aggregate drops 10–15% by 64K–128K.

TTFT at C1 (s), baseline → V1: 2.3 → 2.3 (16K), 4.6 → 4.8 (32K),
9.7 → 10.3 (64K), 21.5 → 22.7 (128K).

## What V1 changed

1. **Activation on real rows only** (`deep_gemm_moe.py`). The expert GEMMs use
   a grouped layout that pads every active expert to a 64-row block, and
   DeepGEMM's SM120 kernel has no smaller block. With about one token per
   expert at decode, 80 real rows became 5,120, and the SITU activation and
   FP8 requantization kernels processed all of them. V1 replaces the two with
   one Triton kernel that visits only the routed rows. Mean step time:
   22.5 → 19.7 ms at C1, 67.2 → 45.9 ms at C8. It matches the stock kernels on
   every scale and on 99.995% of FP8 values; the rest differ by one FP8 step
   from last-bit differences in `tanh` (`tests/test_situ_rows.py`).
2. **Routing kernel hardened against non-finite rows** (`moe.py`). The stock
   kernel returns out-of-range expert ids for a NaN row (id 5110 of 512 in
   `tests/test_routing.py`), which turns any garbage row into an illegal
   memory access. V1 always returns ten distinct valid ids and is identical on
   finite rows. This is what lets adaptive verification start.
3. **Adaptive verification** (`attention.py`, flag). vLLM 0.32 can size each
   request's verification from the drafter's confidence head and measured
   step cost, but ERNIE's DSA backends refuse it on SM120. V1 declares the two
   capabilities the SM120 path already has (per-token indexer rows built from
   device lengths, and variable-length decode graphs). On prose, positions
   4–7 of a 7-token draft are accepted under 10% of the time, so most of the
   verify work was wasted: step time at C8 goes 41 → 33 ms for 7% fewer
   tokens per step.
4. **Memory layout for 524K.** All 16 KV-cache groups share one 378 KB block
   size. At 524K the baseline layout needs 10.6 GiB per card: 3.9 GiB of
   compressor state that scales with the prefill chunk, 3.9 GiB of HCA
   compressed cache that is 96% padding, and 3.1 GiB of real DSA data.
   `--max-num-batched-tokens 2048` cuts the need to 7.54 GiB and the profiled
   activation peak from 4.49 to 3.29 GiB, so 524,288 fits with
   `--kv-cache-memory 8391956153`. Peak use through a 498K-token prefill was
   97,027 of 97,887 MiB.

Tried and dropped:

- **vLLM custom all-reduce.** Still faults on this machine (it starts, then
  dies on the first request), with `iommu=pt` set and P2P reported OK. NCCL
  all-reduce is about 21% of a C8 step, so this is the largest known cost left.
- **Smaller expert block.** Capping DeepGEMM's block at 32 rows fails its
  SM120 heuristic (`not candidates.empty()`); the heuristic is compiled into
  the extension.
- **`--gpu-memory-utilization` instead of an explicit KV budget.** vLLM
  under-counts graph and prefill memory here and over-allocates KV. The
  failure is an illegal memory access during a long prefill, not a clean
  out-of-memory error.

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

# 3. SparkDeck deployment (replace <repo> in the mount paths first)
curl -X POST http://localhost:7878/api/v1/deployments \
  -H 'Origin: http://localhost:7878' -H 'Content-Type: application/json' \
  -d @recipe/sparkdeck-v1-tp2.json        # or sparkdeck-baseline-tp2.json
```

V1 mounts three files into the container (the baseline mounts only
`patch/sm120/.../attention.py`):

```
patch/v1/vllm_ernie/models/ernie5_5/nvidia/attention.py
  -> /opt/vLLM-ERNIE/src/vllm_ernie/models/ernie5_5/nvidia/attention.py
patch/v1/vllm_ernie/models/ernie5_5/nvidia/moe.py
  -> /opt/vLLM-ERNIE/src/vllm_ernie/models/ernie5_5/nvidia/moe.py
patch/v1/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
  -> /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
```

V1 flags (`recipe/sparkdeck-v1-tp2.json`):

```
vllm serve baidu/ERNIE-5.5-Flash-Preview --revision ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe
  --trust-remote-code -tp 2 --disable-custom-all-reduce
  --kv-cache-dtype fp8 --kv-cache-memory 8391956153
  --max-model-len 524288 --max-num-seqs 8 --max-num-batched-tokens 2048
  --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

The baseline differs in three places: `--kv-cache-memory 6264723129`,
`--max-model-len 140000`, and no `--max-num-batched-tokens` or
`enable_adaptive_verification`.

Environment: `HF_HUB_OFFLINE=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`. Startup
takes about 5 minutes (1 minute of weights once cached, the rest is graph
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

Raw results are in `bench/results/`, startup logs in `docs/logs/`, diffs
against the stock files in `patch/diffs/`. Kernel tests run in the image:

```bash
docker run --rm --gpus '"device=0"' -v $PWD/tests:/tests:ro \
  -v $PWD/patch/v1/vllm_ernie/models/ernie5_5/nvidia/moe.py:/opt/vLLM-ERNIE/src/vllm_ernie/models/ernie5_5/nvidia/moe.py:ro \
  -v $PWD/patch/v1/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:ro \
  --entrypoint python3 vllm/vllm-openai:ernie-5.5-flash /tests/test_routing.py
```
