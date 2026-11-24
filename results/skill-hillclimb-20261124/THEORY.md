# Hy4 Flash (SPARK-4Bit-Fast GGUF) on GB10: ceilings

Measured GB10 on this box (2026-11-22, Inkling run): 244 GB/s device copy, 84.8 BF16 TFLOPS dense GEMM.

Weights, from GGUF tensor offsets (`ggufsize.py`): routed experts 81.42 GiB (512 experts, top-16, 42 MoE layers),
attention 3.05 GiB, other dense 0.58, shared experts 0.32, LM head 0.19 (NVFP4), token embd 0.37 (1 row read/token),
PLE/Engram table 62.70 GiB on SSD (a few rows per token, lazily mmapped).

## Decode (bandwidth bound)
- No speculation: experts 81.42 x 16/512 = 2.54 GiB + dense parts 4.14 GiB = 6.68 GiB = 7.17 GB/token -> **34.0 tok/s**.
- MTP depth 4, acceptance 61% (recipe, temp 0) -> ~3.44 tokens/step.
  Per step: dense parts 4.14 GiB once + experts for 5 rows. Independent routing gives 512 x (1 - (1 - 16/512)^5) = 75 unique
  experts/layer -> 11.95 GiB, plus 4 sequential MTP passes (~0.35 GiB each) -> ~17.5 GiB = 18.8 GB/step = 77 ms
  -> **~45 tok/s if routing were independent**. Rows of one block share experts, so the real ceiling is higher;
  measured per-step time is the useful number. Recipe v2: 40.6 tok/s at temp 0 (~90% of the independent bound).
- Levers that move it: fewer wasted draft rows (draft confidence cutoff), better depth/acceptance trade.

## Prefill (compute bound, but measured as launch/memory bound)
Active ~12B params -> ~24 GFLOP/token; 0.5 x 84.8 TFLOPS -> ~1.8K tok/s practical. Recipe: 765 tok/s at 8K (43%).
Recipe's nsys trace: GPU 96% busy at ~50 W, many small memory-bound kernels (64 tokens/expert/ubatch). Kernel work, not flags.

## Startup
87 GiB GPU-resident at the NVMe's ~3.8 GB/s DirectIO = ~23 s floor. Recipe v2 already at 23 s.
