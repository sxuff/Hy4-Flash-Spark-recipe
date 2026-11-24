# Hy4 Flash on GX10: inference-optimisation skill run (2026-11-24)

Skill: sybil-solutions/inference-optimisation-skill @ fda968e (0xSero). Harness: the skill's bench.py, unchanged.
Targets: c1 decode (primary), prefill 4K/32K, startup. Single stream (recipe runs --parallel 1, so cN is skipped).
Constraints: lossless only; 262144 context; keep >= 6 GiB free (watchdog cap 113 GiB host used) and no swap growth.
Baseline: recipe v2 (`~/.hermes/hy4-single-spark/serve.py`, = github sxuff/Hy4-Flash-Spark-recipe 8afd414):
MTP depth 4, DirectIO load, ubatch 2048, whole-tile FA, PDL. Decode panel at temperature 0, no_think.
Prior work (do not rerun without a new reason): load mode, ubatch 1024/2048/4096, MMQ tiles, threads 10,
MTP depth 1-4, FA whole tiles, PDL/graph-opt, cache-ram. See TESTING-AND-RESULTS.md in the recipe.
Not tried before: --spec-draft-p-min (default 0.00), MTP depth > 4, CUDA-graph eligibility under MTP verify.

## Queue 1: ctl-a pmin50 ctl-b pmin75 ctl-c d5 ctl-d diag-graphs
- ctl-a/b/c/d: c1 31.43 / 31.83 / 32.09 / 32.31 (per prompt 25-43; last-prompt draft acceptance 36.6%, mean len 2.46,
  identical across controls: greedy is deterministic here). Prefill 4K 667-837 (4 s TTFT, noisy), 32K 792-820. Startup 23-26 s.
  ctl-a prefill lower: first arm after Inkling stop (cold Engram pages).
- pmin50 (--spec-draft-p-min 0.5): ACCEPTED. c1 33.81, +6.2% over the better neighbour (ctl-b), all 6 prompts up. Acceptance 48%.
- pmin75 (--spec-draft-p-min 0.75): ACCEPTED (best). c1 35.16, +9.6% over the better neighbour (ctl-c). Acceptance 72%.
  Prefill unchanged (835 / 815). Repeat queued (single surprising win).
- d5 (MTP depth 5, no cutoff): REJECTED. c1 28.80 (-10%), acceptance 30%: extra draft rows mostly wasted.
- diag-graphs (GGML_CUDA_GRAPH_DEBUG=1): CUDA graphs active under MTP verify ("graphs reused = 1048", 0 changed-size
  reallocs). Debug lines need verbose logging; the reuse counter answers it. Lever 2 (decode graphs) already in place.

## Queue 2: ctl-e pmin90 ctl-f d6p75 ctl-g pmin75r ctl-h eq:ctl-h
- ctl-e/f/g/h: c1 32.07 / 32.25 / 31.84 / 31.63. Eight controls overall: 31.43-32.31 (mean 31.93, +-1.5%).
- pmin90: REJECTED, inside noise. c1 31.97. Acceptance 82% but too few drafts: fast prompts lose (code 42.1 -> 39.8).
- d6p75 (depth 6 + p-min 0.75, one change from the best): REJECTED vs best. c1 34.05 (+5.6% over ctl-f, -3% vs pmin75).
- pmin75r (repeat of the best): CONFIRMED. c1 35.00 vs 35.16 first run; identical draft stats (greedy deterministic).
  pmin75 vs 8 controls: +9.6% on the skill panel. Beats both neighbours both times.
- eq:ctl-h: greedy reference outputs captured from the control server (logs/eq-ctl-h.json).

## Queue 3 (qualification of pmin75): long:best-long eq:best ctl-t07 best-t07 ctl-t07b best-t07b ctl-t07c
- long:best-long (pmin75 + needle): c1 34.99 (third run). Needle: 68,768 prompt tokens (bench's 100K target; Hy4's tokenizer
  packs ~0.69 tokens/word-estimate), retrieval 3/3, prefill 745 tok/s, decode 44.77 tok/s at 69K context. PASS.
- eq:best vs eq:ctl-h (greedy, same 6 prompts): 1/6 byte-identical. First differences are near-tie picks
  ("tests:" vs "tests.", "rings in an old oak stump" vs "rings of an old tree", two equivalent table phrasings),
  i.e. verify-batch-shape numerics (p-min changes how many rows each verify step has). Not yet proven lossless:
  queued an MTP-off reference (d0 + eq:d0) to test whether plain depth 4 diverges from no-MTP greedy just as often.
- Temperature 0.7 (sampling noise is larger: text differs every run):
  ctl-t07 32.55, best-t07 35.01, ctl-t07b 34.48, best-t07b 33.71, ctl-t07c 34.55.
  pmin75 mean 34.36 vs ctl mean 33.86 (+1.5%): INSIDE NOISE at 0.7. best-t07b lost to its left neighbour.
  Per prompt (2 vs 2 runs): story +17%, table +7%, summary +8%, bash +4%, TCP ~0, LRU-cache code ~-7%.
  More 0.7 samples queued (q5), plus p-min 0.5 at 0.7.

## Queue 4: d0 eq:d0 long:best-l131:190000 long:best-l245:356000
## Queue 5: pmin50-t07 ctl-t07d best-t07c ctl-t07e pmin50-t07b ctl-t07f
- d0 (MTP off): c1 26.16 on every prompt (no drafting, no per-prompt spread) = 77% of the 34 tok/s no-spec ceiling.
  MTP depth 4 adds +22%; depth 4 + p-min 0.75 adds +35% over no MTP.
- Greedy equivalence vs MTP off (eq:d0): the recipe's own MTP (ctl-h) diverges from no-MTP greedy on 5/6 prompts
  (first difference at chars 76-899); pmin75 diverges at the same points on 3 prompts and nearby on the rest.
  Drafting in this build is not bit-exact vs no drafting (verify batch shapes); p-min adds no new class of deviation.
  Verdict: as faithful as the recipe's current MTP. PASS (labelled as such, not "bit-identical").
- long:best-l131 (pmin75): 130,608 tokens, retrieval 3/3, prefill 662 tok/s, decode 40.19 at 131K. PASS.
- long:best-l245 (pmin75): 244,793 tokens (93% of 262144), retrieval 3/3, prefill 523 tok/s, decode 38.15. PASS.
- Queue 5 (temp 0.7): pmin50-t07 34.87, ctl-t07d 30.70, best-t07c 35.00, ctl-t07e 32.41, pmin50-t07b 33.85, ctl-t07f 32.58.

## Pooled result (time-weighted: total tokens / total decode time over the 6 x 512-token panel)
| | control | p-min 0.75 | gain | per-run ranges |
| temp 0   | 32.31 (8 runs, 31.86-32.63) | 34.58 (5 runs, 34.43-34.78) | +7.0% | no overlap |
| temp 0.7 | 31.81 (6 runs, 30.98-32.78) | 33.98 (3 runs, 33.51-34.32) | +6.8% | no overlap |
p-min 0.5 at temp 0.7: 33.51 (+5.3%). Per-run medians were noisier (prompt spread 25-43 tok/s); time-weighted is the metric.

## STATUS (plateau: pmin90, d6p75 and d5 inside noise or worse after the win)
Best: recipe v2 + `--spec-draft-p-min 0.75`. Decode +7% (temp 0 and 0.7), prefill/startup/memory unchanged,
long context 3/3 at 69K/131K/245K, greedy as faithful as the recipe's MTP.
Limits: no-MTP decode at 77% of the bandwidth ceiling; MTP acceptance is content-bound (code ~ high, prose low);
prefill at ~43% of a practical compute ceiling, bound by small MoE kernels (recipe nsys trace); startup at NVMe floor.
Remaining levers (ranked): fused MoE / attention prefill kernels (large); pruned draft LM head (medium, code);
per-content adaptive draft length (medium, code).
