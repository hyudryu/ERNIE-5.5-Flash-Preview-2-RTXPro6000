# ERNIE-5.5-Flash-Preview on 2× RTX PRO 6000 Blackwell

Recipe, SM120 patch and benchmarks for serving
[baidu/ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)
(280B-parameter MoE, 14B active, 1M context) with vLLM on two 96 GB RTX PRO 6000
Blackwell workstation cards (SM120).

Every number below is **TP2, FP8 KV cache, MXFP4 experts / FP8 dense as shipped
(no extra quantization)**, DSpark speculative decoding, thinking on (the model
default).

## Status

| | Baseline | V1 | V2 |
| --- | --- | --- | --- |
| vLLM | 0.32.0 + [vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) `ba322a5` | same | same |
| Patch | SM120 attention port (1 file) | 3 files (`patch/v1`) | 4 files (`patch/v2`) |
| Layout | TP2, FP8 KV | TP2, FP8 KV | TP2, FP8 KV |
| DSpark draft tokens | 7, probabilistic | 7, probabilistic, adaptive verification | same as V1 |
| Max sequences | 8 | 8 | 8 |
| Prefill chunk | 8,192 | 2,048 | 1,024 |
| Context per session | 140,000 | 524,288 | 300,000 |
| KV pool | 140,304 tokens (5.83 GiB per card) | 543,803 tokens (7.82 GiB per card) | **1,213,175 tokens** (9.02 GiB per card) |
| Full-length sessions in the pool | 1.0 | 1.04 | **4.04** |
| Mean decode, C1 / C2 / C4 / C8 | 165 / 279 / 389 / 535 tok/s | 228 / 297 / 574 / 841 tok/s | **233 / 321 / 613 / 864 tok/s** |
| Mean step time, C1 / C2 / C4 / C8 | 22.5 / 30.4 / 45.3 / 67.2 ms | 19.5 / 25.5 / 30.6 / 40.8 ms | **18.4 / 24.2 / 29.3 / 39.4 ms** |

The stock plugin does not start on SM120. The baseline is the smallest patch
that makes Baidu's recommended flags run correctly on these cards, with
nothing tuned for speed. V1 adds exact kernel-path fixes, the memory layout
for a 524K context, and vLLM's adaptive verification. V2 removes the padding
in the KV-cache layout, which more than doubles the pool on the same FP8
cache, and caps each session at 300K so the pool holds four of them. Decode is
within a few percent of V1. Nothing is quantized beyond what the checkpoint
ships.

## Results

Quick runs: one pass per cell, 1,024 output tokens, temperature 0.
`ms/step` is the wall time of one decode step (draft + verify) and `tok/step`
is tokens emitted per step, both from `/metrics` deltas. Run-to-run variation
is about 5%, mostly from draft acceptance (the drafter samples).

### Decode, no context (C1–C8)

| Workload | C | Baseline tok/s | V1 tok/s | V2 tok/s | Baseline ms/step | V1 ms/step | V2 ms/step | Baseline tok/step | V1 tok/step | V2 tok/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prose | 1 | 102.9 | 125.6 | 134.9 | 21.8 | 17.7 | 16.6 | 2.25 | 2.23 | 2.23 |
| prose | 2 | 156.9 | 205.6 | 228.1 | 28.5 | 20.8 | 19.9 | 2.24 | 2.14 | 2.27 |
| prose | 4 | 216.4 | 337.9 | 366.0 | 42.4 | 25.8 | 24.5 | 2.29 | 2.18 | 2.24 |
| prose | 8 | 294.8 | 507.5 | 538.7 | 61.7 | 33.0 | 31.5 | 2.27 | 2.09 | 2.12 |
| coding | 1 | 163.9 | 189.6 | 180.9 | 23.0 | 20.4 | 19.1 | 3.77 | 3.87 | 3.45 |
| coding | 2 | 232.2 | 252.9 | 286.4 | 32.3 | 28.8 | 27.0 | 3.75 | 3.64 | 3.87 |
| coding | 4 | 296.5 | 429.9 | 468.4 | 50.3 | 34.9 | 33.6 | 3.73 | 3.75 | 3.93 |
| coding | 8 | 438.8 | 613.2 | 630.1 | 76.6 | 51.4 | 48.1 | 4.20 | 3.94 | 3.79 |
| counting | 1 | 228.3 | 370.0 | 384.2 | 22.6 | 20.4 | 19.7 | 5.17 | 7.55 | 7.55 |
| counting | 2 | 448.9 | 431.7 | 448.6 | 30.5 | 26.9 | 25.6 | 6.84 | 5.81 | 5.75 |
| counting | 4 | 655.2 | 954.8 | 1004.7 | 43.2 | 31.2 | 29.8 | 7.07 | 7.44 | 7.50 |
| counting | 8 | 871.6 | 1401.2 | 1423.5 | 63.2 | 38.0 | 38.7 | 6.88 | 6.66 | 6.89 |

Aggregate tok/s. The counting rows move with draft acceptance from run to run
(60% to 95% at C1), so read `ms/step` for the kernel-level change there.

| Mean of the three workloads | Baseline | Kernel fix only | V1 | V2 |
| --- | --- | --- | --- | --- |
| C1 tok/s (ms/step) | 165.0 (22.5) | 226.8 (19.7) | 228.4 (19.5) | 233.3 (18.4) |
| C2 tok/s (ms/step) | 279.3 (30.4) | 312.7 (25.8) | 296.8 (25.5) | 321.0 (24.2) |
| C4 tok/s (ms/step) | 389.4 (45.3) | 550.1 (33.8) | 574.2 (30.6) | 613.0 (29.3) |
| C8 tok/s (ms/step) | 535.1 (67.2) | 788.4 (45.9) | 840.6 (40.8) | 864.1 (39.4) |

"Kernel fix only" is V1 without adaptive verification, on the baseline's
memory layout. V2's decode path is V1's plus the PCIe IPC all-reduce; the
2–5% lower step time is at the edge of run-to-run variation.

These cards are held at 80 °C by a power governor on this machine. After an
hour of continuous load GPU 1 was cut from 450 W to 380 W and the same V2 run
measured 19.1 / 25.7 / 31.3 / 43.1 ms per step, about 8% slower at C8. The
numbers in the tables were taken after a short cooldown, at the full 450 W.

### Prefill

Unique documents, so nothing is served from the prefix cache. Warm rounds.

| Prompt tokens | Baseline TTFT (s) | V1 TTFT (s) | V2 TTFT (s) | Baseline tok/s | V1 tok/s | V2 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1K | 0.20 | 0.18 | 0.20 | 5,130 | 5,770 | 5,200 |
| 4K | 0.60 | 0.59 | 0.65 | 6,860 | 6,970 | 6,320 |
| 16K | 2.31 | 2.30 | 2.44 | 7,120 | 7,090 | 6,690 |
| 32K | 4.60–5.44 | 4.74 | 4.93 | 6,020–7,130 | 6,930 | 6,630 |
| 64K | 9.44 | 9.79–10.5 | 10.3 | 6,920 | 6,190–6,680 | 6,380 |
| 128K | 20.5 | 20.2–23.4 | 22.2 | 6,370 | 5,590–6,460 | 5,890 |
| 262K | does not fit | 49.6 | 52.0 | — | 5,250 | 5,020 |
| 289K | does not fit | not run | 61.5 | — | — | 4,690 |
| 498K | does not fit | 118.9 | over the 300K cap | — | 4,190 | — |

Prefill speed is about the same up to 32K and up to 10% lower at 64K–128K:
V1 prefills in 2,048-token chunks to free the memory that makes the 524K
context possible, and V2 uses 1,024-token chunks to shrink the compressor's
scratch state further. That costs about 5% at 16K–32K and nothing consistent
beyond that. The fused activation kernel on its own makes prefill about
4% faster (7,500 / 7,140 / 6,640 tok/s at 16K / 64K / 128K with 8,192-token
chunks).

### Decode with context

Prose after a filler document of the given size, 512 output tokens. Aggregate
tok/s while all requests decode; "—" means the cell does not fit the KV pool.

| Context | Baseline C1 | V1 C1 | V2 C1 | Baseline C2 | V1 C2 | V2 C2 | Baseline C4 | V1 C4 | V2 C4 | Baseline C8 | V1 C8 | V2 C8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 102.9 | 125.6 | 134.9 | 156.9 | 205.6 | 228.1 | 216.4 | 337.9 | 366.0 | 294.8 | 507.5 | 538.7 |
| 16K | 108.2 | 125.6 | 136.2 | 167.4 | 208.1 | 211.9 | 216.9 | 337.3 | 331.6 | n/a | 493.1 | 486.3 |
| 32K | 109.9 | 125.0 | 134.5 | 160.7 | 198.9 | not run | 208.6 | 308.7 | 331.7 | — | 442.1 | not run |
| 64K | 110.2 | 131.5 | 134.7 | 162.3 | 191.1 | not run | — | 295.3 | 350.0 | — | n/a | not run |
| 128K | 105.7 | 123.1 | 141.4 | — | 181.4 | not run | — | 316.2 | n/a | — | — | n/a |

"n/a": the requests did not all decode at the same time, so there is no
steady aggregate. "—": the cell does not fit the pool. "not run": V2 used a
reduced grid (all of 16K, then C1 and C4, plus C8 at 128K). At C1 decode
speed does not depend on context length. At higher concurrency the V1
aggregate drops 10–15% by 64K–128K and V2's drops 5–10% at C4.
Eight concurrent 128K prompts (1.04M tokens) fit the V2 pool and complete;
V1 could hold four.

TTFT at C1 (s), baseline → V1 → V2: 2.3 → 2.3 → 2.5 (16K), 4.6 → 4.8 → 5.1 (32K),
9.7 → 10.3 → 10.4 (64K), 21.5 → 22.7 → 23.3 (128K).

## What V2 changed

V2 is about the KV pool. The cache is still FP8 (`--kv-cache-dtype fp8`), the
same bytes per token as before; nothing is quantized. The pool grew because
most of it was padding.

| Step | Pool (tokens) | KV memory per card |
| --- | --- | --- |
| V1 | 543,803 | 7.82 GiB |
| 1. All 37 HCA compressed caches in one KV group | 753,699 | 7.82 GiB |
| 2. Prefill chunk 2,048 → 1,024 tokens | 900,364 | 7.82 GiB |
| 3. Block stride 377,856 → 335,872 bytes | 1,051,704 | 7.82 GiB |
| 4. KV budget raised, stress-tested | **1,213,175** | 9.02 GiB |

1. **One group for the compressed caches** (`kv_cache_utils.py`). vLLM packs
   layers with different page sizes into groups that share one block size.
   Its packer split the 37 HCA compressed caches into five groups to mirror
   the sliding-window caches, but a compressed cache needs only 1.7 KB per
   256-token block, so each group filled under 4% of a 378 KB block and a
   524K session spent 3.6 GiB on it. V2 puts a single-page-size bucket into
   the fewest groups that fit the block: one group, 64 KB per block.
2. **1,024-token prefill chunk** (flag). The compressor keeps scratch state
   for every token in flight (twice the chunk), so halving the chunk halves
   that fixed cost. Prefill is about 5% slower at 16K–32K and within
   run-to-run variation of V1 from 64K up.
3. **Stride alignment for the FP8 layout** (`attention.py`). The plugin
   aligns the shared block stride to 9,216 bytes so that bf16 caches can be
   viewed as flat rows. On SM120 the caches are packed `fp8_ds_mla` pages
   (584- and 656-byte entries that this alignment never matched), and only
   the indexer's 512-byte rule applies. The block shrinks from 377,856 to
   335,872 bytes and the DSA group now fills 90% of it instead of 80%.
4. **KV budget.** With the layout fixed, 9.02 GiB per card passes one 282K
   prefill followed by four concurrent 282K-token prompts with decode on top
   (`tests/stress_long_context.py`); every answer retrieves a planted needle.
5. **FlashInfer PCIe IPC all-reduce** (env). vLLM 0.32 ships an opt-in
   all-reduce for PCIe-only nodes. Unlike vLLM's custom all-reduce it runs on
   this machine. It replaces NCCL for the bf16 hidden-state reductions and is
   worth about 2% of a step at C1 and nothing measurable at C8.

The layout changes move where pages sit, not what is stored.
`tests/test_layout_regression.py` checks needle retrieval and the first
generated token at 100, 7K, 47K and 153K prompt tokens against the V1
layout; all pass, with the same run-to-run logprob noise as two runs of the
same build.

Decode is where V1 left it, and the profile says why: at C8 a step routes 640
rows over 512 experts, so nearly every expert weight is read each layer and
the step is bound by memory bandwidth, not by kernel overhead. At C1, 80
routed rows still read 80 experts' weights per layer.

Tried and dropped in V2:

- **Larger compressed-cache blocks.** A 1,024-token HCA block would add
  another 17% to the pool, but FlashInfer's SM120 prefill kernel is compiled
  for 2 or 64 compressed entries per block only
  (`Unsupported sparse-MLA prefill configuration ... extra_page_block_size=8`).
  64 entries (8,192 tokens) would work in principle, but prefix-cache hits
  land on the least common multiple of the group block sizes, so every cached
  turn would re-prefill up to 8,191 tokens. With 256-token blocks a follow-up
  turn on a 21.7K-token chat recomputes 370–560 tokens and answers in 0.15 s
  (`tests/prefix_hit.py`). The switch is left in as
  `ERNIE_HCA_BLOCK_TOKENS`, default 256; other values are not validated.
- **Splitting large blocks into kernel blocks.** vLLM can do this, but it
  requires the stride to be a multiple of the kernel page, which pushes the
  shared block to 1.13 MB and the pool down to 397K tokens.
- **Custom router GEMM.** The fp32 router GEMM is 9% of a C1 step, but its
  16 MB weight read is most of that; a hand-written kernel could recover
  about 2% and would reorder fp32 sums, so it was not built.

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

Where the time goes now (C1, 17–19 ms per step): about half is FP8/FP4 GEMMs
that already run near the memory-bandwidth limit of the weights they read,
10% is DeepGEMM's split-K reductions, 9% cross-GPU all-reduce, 9% the fp32
router GEMM, 7% attention. No single remaining item is worth more than about
3%, and the last probe below came out inside the noise, so V1 is where the
exact, flag-and-overlay work stops paying. The larger items left need a
DeepGEMM rebuild (a block smaller than 64 rows) or a working custom
all-reduce.

Tried and dropped:

- **vLLM custom all-reduce.** Still faults on this machine (it starts, then
  dies on the first request), with `iommu=pt` set and P2P reported OK. NCCL
  all-reduce is about 21% of a C8 step, so this is the largest known cost left.
- **No auxiliary-stream overlap** (`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0`).
  Mean step time 19.0 / 24.5 / 29.8 / 40.3 ms at C1 / C2 / C4 / C8 against
  V1's 19.5 / 25.5 / 30.6 / 40.8, which is inside run-to-run variation.
- **Merging the two MoE all-reduces per layer.** The profile puts the saving
  at about 1% of a step, so it was not built.
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

# 3. Serve on port 8000 (plain docker run; DRY_RUN=1 prints the command)
recipe/run.sh v2          # or: v1, baseline
```

`recipe/run.sh` bind-mounts the patch files of the chosen version over the
ones in the image and passes the flags below. V2 mounts four files:

```
patch/v2/vllm_ernie/models/ernie5_5/nvidia/attention.py
  -> /opt/vLLM-ERNIE/src/vllm_ernie/models/ernie5_5/nvidia/attention.py
patch/v2/vllm_ernie/models/ernie5_5/nvidia/moe.py
  -> /opt/vLLM-ERNIE/src/vllm_ernie/models/ernie5_5/nvidia/moe.py
patch/v2/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
  -> /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
patch/v2/vllm/v1/core/kv_cache_utils.py
  -> /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
```

V1 mounts the first three from `patch/v1`; the baseline mounts only
`patch/sm120/.../attention.py`.

V2 flags:

```
vllm serve baidu/ERNIE-5.5-Flash-Preview --revision ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe
  --trust-remote-code -tp 2 --disable-custom-all-reduce
  --kv-cache-dtype fp8 --kv-cache-memory 9680446000
  --max-model-len 300000 --max-num-seqs 8 --max-num-batched-tokens 1024
  --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

Environment: `HF_HUB_OFFLINE=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`, and for V2
`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`. Startup takes about 5 minutes
(1 minute of weights once cached, the rest is graph capture and kernel
warm-up).

| Flag | Baseline | V1 | V2 |
| --- | --- | --- | --- |
| `--max-model-len` | 140000 | 524288 | 300000 |
| `--kv-cache-memory` | 6264723129 | 8391956153 | 9680446000 |
| `--max-num-batched-tokens` | default (8192) | 2048 | 1024 |
| `enable_adaptive_verification` | no | yes | yes |

`--max-model-len` is the cap per session, not the pool: the V2 pool holds four
300K sessions, or more shorter ones (up to `--max-num-seqs 8` at once). A
larger cap has not been tested with the V2 memory budget. If a long prefill
ever faults, drop `--kv-cache-memory` to 9036201000 (pool 1,132,440 tokens),
which leaves 0.6 GiB more headroom.

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

Checks against a running server (port 8000):

```bash
python3 tests/test_layout_regression.py save ref.json   # on the reference build
python3 tests/test_layout_regression.py cmp ref.json    # on the build under test
python3 tests/stress_long_context.py "1x247000,4x247000"  # prompt words; ~282K tokens each
python3 tests/prefix_hit.py                              # cached tokens per chat turn
```
