# Recipe v2.1: MTP draft confidence cutoff (`--spec-draft-p-min 0.75`)

A hill-climb of recipe v2 with [0xSero](https://github.com/0xSero)'s
[inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) (commit `fda968e`),
on the same ASUS GX10 (NVIDIA GB10), 2026-11-24. The skill's method ran as written: ceilings first, one change per run,
a control run on both sides of every candidate, a ledger of every run including the failures, then long-context and
output gates before adopting the winner. Its engine-agnostic benchmark (`bench.py`) ran unchanged.

## Result

llama.cpp's MTP drafting defaults to `--spec-draft-p-min 0.00` in this build, so every round drafts all four tokens.
With a 0.75 cutoff the draft stops at the first token the MTP head is less than 75% sure of, and verify steps carry
fewer rows that get thrown away (draft acceptance on the last panel prompt: 37% → 72%).

Time-weighted decode (total tokens / total decode time over the skill's 6-prompt, 512-token panel, single stream):

| | v2 (control) | v2.1 (`--spec-draft-p-min 0.75`) | gain | runs |
|---|---:|---:|---:|---|
| temperature 0 | 32.31 tok/s (31.86–32.63) | **34.58 tok/s** (34.43–34.78) | **+7.0%** | 8 controls, 5 candidates, no overlap |
| temperature 0.7 | 31.81 tok/s (30.98–32.78) | **33.98 tok/s** (33.51–34.32) | **+6.8%** | 6 controls, 3 candidates, no overlap |

The slowest v2.1 run beats the fastest control at both temperatures. Prefill (~800 tok/s at 3K and 23K), startup
(23–26 s) and memory are unchanged. These are the skill's panel and harness, not comparable with `bench/tune.py`'s
four 192-token workloads behind the result card.

**On `bench/tune.py`** (arms `results/r9-*`, the same day, v2 on both sides): temp 0 is unchanged (39.67 → 39.76
tok/s), because that panel's code, math and JSON already accepted 61% of the drafts. Its temp 0.7 (−3.9%) and 0.9
(+14%) cells each sample one fixed text, so they move both ways. Prefill, load, the prompt cache, the 60K needle and
the smokes match v2. The gain therefore depends on the text: about +7% on mixed, prose-heavy prompts, none on
already-predictable ones. [TESTING-AND-RESULTS.md](../../TESTING-AND-RESULTS.md#the-same-change-on-benchtunepy-r9--2026-11-24)

## Gates

- **Long context**, pmin 0.75, a document with three planted codes: 68,768 tokens (3/3, decode 44.8 tok/s), 130,608
  tokens (3/3, 40.2 tok/s) and 244,793 tokens (93% of the 262,144 context; 3/3, prefill 523 tok/s, decode 38.2 tok/s).
- **Outputs**: greedy text is not byte-identical to v2 (1 of 6 panel prompts), and the first differences are near-tie
  picks ("tests:" vs "tests.", "rings in an old oak stump" vs "rings of an old tree"). Against a run with MTP off,
  v2's own MTP diverges on 5 of 6 prompts too, at characters 76–899, and v2.1 diverges at the same points on 3 of them:
  drafting in this build is not bit-exact against no drafting (the verify batch shape changes the kernels), and the
  cutoff adds no new kind of difference. Only hashes are kept here (`eq/`), not the generated text.

## Everything tried (one change from the current best each time)

| Run | Change | c1 median | Verdict |
|---|---|---:|---|
| ctl ×8 | v2 as published | 31.43–32.31 | baseline (±1.5%) |
| pmin50 | `--spec-draft-p-min 0.5` | 33.81 | better, but below 0.75 |
| **pmin75** | `--spec-draft-p-min 0.75` | **35.16, repeat 35.00** | **adopted** |
| pmin90 | `--spec-draft-p-min 0.9` | 31.97 | inside noise: too few drafts, the code prompt slows |
| d5 | MTP depth 5, no cutoff | 28.80 | −10%, acceptance 30% |
| d6p75 | depth 6 with the 0.75 cutoff | 34.05 | −3% against depth 4 with the cutoff |
| d0 | MTP off | 26.16 | reference: 77% of the 34 tok/s no-drafting ceiling |
| diag-graphs | `GGML_CUDA_GRAPH_DEBUG=1` | 32.50 | CUDA graphs active under MTP verify (1,048 reuses, 0 changed-size reallocs) |

## Where it stops

[`THEORY.md`](THEORY.md): 7.2 GB read per token without drafting (34 tok/s at the 244 GB/s measured on this box); with
MTP at depth 4, ~45 tok/s if routing were independent across the verified rows. Prefill sits at ~43% of a practical
compute ceiling, bound by small MoE kernels (the v2 Nsight trace), and the 23 s load is at the NVMe's DirectIO speed.
What is left is kernel work (fused MoE and attention prefill kernels, a pruned draft LM head), not flags.

## Files

- [`PLAN.md`](PLAN.md): the timestamped log of every queue, run and verdict.
- [`ledger.jsonl`](ledger.jsonl): one line per run from the skill's `bench.py` (c1 per prompt, prefill, startup, peak memory).
- [`THEORY.md`](THEORY.md): the ceilings, from the GGUF's tensor sizes (`tools/ggufsize.py`).
- `tools/`: the run as it was done, with home shown as `~`. `hc_serve.py` launches the recipe's own `serve.py`
  `materialize()` plus one extra flag; `gen.py` writes one `bench.py` config per run; `hostmem.sh` is the memory probe
  (GB10 reports per-process GPU memory as `[N/A]`, so it reads host-wide MemAvailable and trips on swap growth);
  `queue3.sh` runs a queue serially behind a lock; `eq.py` captures and compares greedy outputs.
- `eq/`: SHA-256 prefixes and lengths of the greedy outputs (v2 `ctl-h`, v2.1 `best`, MTP off `d0`).

To reproduce with the recipe's own harness, add `"extra": ["--spec-draft-p-min", "0.75"]` to a `bench/tune.py` arm.
