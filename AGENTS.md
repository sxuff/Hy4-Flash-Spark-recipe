# AGENTS.md

This repository serves Hy4 Flash (AngelSlim `SPARK-4Bit-Fast` GGUF) with llama.cpp on one DGX Spark / GB10.

- **Commands:** `bash build.sh`, then `python3 download.py`, then `python3 serve.py [--cache-ram 4096]`.
  `python3 serve.py --dry-run` prints the exact launch without running it.
- **Source of truth for flags:** `serve.py`. If you change a flag, re-measure with `bench/tune.py`, add
  the arm's receipts under `results/`, then run `python3 tools/make_card.py` and update the README table and
  `TESTING-AND-RESULTS.md`. Card numbers are read from `results/`; never edit the SVGs by hand.
- **Do not:**
  - run Hy4 next to another GPU model; it needs the whole GB10.
  - raise `--cache-ram` above 4096 at 256K context.
  - drop `--override-tensor` for `per_layer_token_embd`; the 62.7 GiB table must stay on the SSD.
  - expose the server beyond loopback without adding an API key.
- **Bench harness:** `bench/tune.py begin` stops the incumbent server named in `bench/qualify.py`
  (`INC`, `OLD`) and arms a 4-hour fallback restore. `arm NAME JSON` runs one configuration. `end` restores
  the incumbent. Edit `INC`/`OLD` for your box. Run it under `systemd-run --user -p MemorySwapMax=0`,
  because the guard requires a no-swap cgroup.
- `bench/qualify.py`'s own `execute()` (the v1 qualification run) writes `RECIPE.md` and `RESULTS.json`
  at the repo root. Both are git-ignored.
