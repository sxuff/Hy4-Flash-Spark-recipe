# Testing and results

Everything here was measured on one ASUS GX10 (NVIDIA GB10, 128 GB unified memory, 1 TB NVMe,
CUDA 13.0) on 2026-11-19 with `bench/tune.py`. Each configuration ("arm") ran on a freshly started
server under a memory guard (kill below 6 GiB available or on any swap growth). Its receipts are
in `results/<arm>/`.

## Method

- **Prefill:** prompts cut from llama.cpp's own source (non-repetitive, so the Engram table sees
  real n-gram variety), wrapped in the chat template, `cache_prompt: false`, streamed. TTFT is
  measured client-side to the first streamed token. Each size runs twice. The second "warm" run
  shows the cost of SSD reads, because the Engram pages are then in page cache.
- **Decode:** four workloads (code, math, prose, structured JSON), 192 generated tokens each,
  `ignore_eos`, median across workloads. The depth comparison repeats this twice at temps 0, 0.7 and 0.9.
- **Prompt reuse:** a 16K conversation plus one follow-up turn; two 16K conversations interleaved
  A → B → A → B; eight 16K conversations, then revisiting the newest two and the oldest.
- **Retrieval:** a code word planted 30% of the way into 60K, 120K and 200K tokens of source, asked for at the end.
- **Smokes, every arm:** arithmetic answer, visible reasoning, image colour recognition, parsed tool call.

## What changed vs v1, and why

| Setting | v1 | v2 | Evidence (arms) |
|---|---|---|---|
| `--load-mode` | mmap | **dio** | Load 321 s → **23 s**: 87 GiB at ~3.8 GB/s vs ~0.28 GB/s page-faulting (`r1-base`, `r3-dio`). Same smokes, MTP acceptance and memory floor. The lazy Engram table is still mmapped. |
| `--ubatch-size` / `--batch-size` | 1024 / 2048 | **2048 / 4096** | Prefill +7% at 2K–32K (`r1-ub2048`). ubatch 4096 was 4% slower and used 2.5 GiB more (`r2-ub4096-auto`). |
| `GGML_CUDA_FA_576_WHOLE_TILES` | 0 | **1** | Same-session A/B, warm: 2K 815→860, 8K 817→849, 32K 745→770 tok/s (`r5-fa-stream` vs `r5-fa-whole`). Only affects dense-attention batches above 16 tokens, so decode is unchanged. |
| `GGML_CUDA_PDL`, `GGML_CUDA_GRAPH_OPT` | unset | **1** | Decode-neutral to slightly positive; recovered the ubatch-2048 decode dip (`r1-ub2048-pdl`). |
| MTP depth (`--spec-draft-n-max`) | 2 | **4** | Decode table below (`r6-262k-d1`…`d4`). v1 picked depth 2 from one run at temp 0.9. |
| `--ctx-size` | 65536 | **262144** | Retrieval passed at 60K, 120K and 200K (`r4-262k-depth`). The KV cache costs ~7.5 GiB more than at 64K. |
| `--cache-ram` | 0 | **4096 recommended** | Conversation switching drops from ~21 s to 0.5 s (`r7-final-*`, `r8-cram4g-stress`). |

## Prefill (final v2 configuration, `r7-final-cram0`)

| Prompt | v1 prefill | v1 TTFT | **v2 prefill** | **v2 TTFT** | v2 warm |
|---|---|---|---|---|---|
| 2K | 664 tok/s | 3.15 s | **749 tok/s** | **2.79 s** | 851 tok/s |
| 8K | 694 tok/s | 11.8 s | **765 tok/s** | **10.7 s** | 842 tok/s |
| 32K | 638 tok/s | 52.3 s | **706 tok/s** | **47.2 s** | 750 tok/s |
| 60K | — | — | 711 tok/s | 84.8 s | |
| 120K | — | — | 635 tok/s | 190 s | |
| 200K | — | — | 542 tok/s* | 369 s | |

\*Measured before whole-tile FA was enabled (`r4-262k-depth`); expect ~3% better now.

## Decode at 256K context (`r6-262k-d*`, mean of two reps)

| MTP depth | temp 0 | temp 0.7 | temp 0.9 |
|---|---|---|---|
| 1 | 34.7 | 34.0 | 33.3 |
| 2 (v1) | 38.5 | 34.7 | 35.5 |
| 3 | 41.0 | 35.1 | 36.7 |
| **4 (v2)** | **40.6** | **39.3** | **36.2** |

Above temp 0 this comparison is noisy. The number of draft tokens changes how the sampler's
random stream is consumed, so each depth samples different text. Depth 4 is never more than
0.6 tok/s behind the best and leads clearly at 0.7. Draft acceptance at depth 4 is 61% (temp 0)
to 51% (temp 0.9).

## Prompt reuse (16K-token conversations)

| Case | `--cache-ram 0` | `--cache-ram 4096` |
|---|---|---|
| Next turn of the same conversation | 0.28 s | 0.28 s |
| Back to conversation A after working in B | 20.7 s (full re-prefill) | **0.48 s** |
| 8 conversations, then revisit the oldest | — | 0.44 s (nothing evicted, ~2.6 GiB cached) |

## Memory envelope (lowest MemAvailable during the arm)

| Configuration | Lowest free |
|---|---|
| v1 (64K, depth 2) | 22.5 GiB |
| v2, no prompt cache, incl. a 60K prompt | 13.1 GiB |
| v2, 200K prompt | 12.9 GiB |
| v2, `--cache-ram 4096`, 8 cached conversations + 120K prompt | **9.7 GiB** |

With the cache capped at 4 GiB, the worst case is about 8 GiB free. At 256K context, 8192 could
fall below 6 GiB.

## Where prefill time goes

From the Nsight Systems trace of an 8K prefill (`results/r3-prof/kern_sum.csv`):

| Share of GPU time | Kernels |
|---|---|
| 22% | Flash attention, 576/512 MLA layers |
| ~29% | NVFP4 MoE expert matmuls |
| ~6.5% | NVFP4 activation quantisation |
| 5% | Q8_0 matmuls |
| ~11% | Hyper-connection pre/post, norms and elementwise ops |
| 2% | Lightning indexer |

The GPU is about 96% busy but draws only about 50 W. That is the signature of many small,
memory-bound kernels: 512 experts with 16 active gives roughly 64 tokens per expert per ubatch.
SSD reads of the Engram table cost only 8–11% of prefill, and the disk was busy 5–9% of the time.
Further prefill gains need kernel work (fused MoE, attention), not flags.

## Tested and rejected

- 10 threads pinned to the Cortex-X925 cores (`r1-ub2048-t10`): no change, because the CPU is not on the critical path.
- `GGML_CUDA_SPARK_MMQ_TILE` adaptive (−1) or 128 (`r2-tile-*`): no gain over 64, slightly slower decode.
- ubatch 4096 (`r2-ub4096-auto`): slower prefill, +2.5 GiB.
- Per-request `speculative.*` parameters: compiled out in this llama.cpp build (`#if 0` in
  `tools/server/server-schema.cpp`), so MTP depth is a launch-time setting.
