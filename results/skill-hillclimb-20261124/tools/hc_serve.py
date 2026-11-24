#!/usr/bin/env python3
"""Hill-climb launcher for Hy4: the recipe's own serve.py materialize() and ENV, unchanged,
plus extra llama-server flags given after '--' (one change per arm)."""
import argparse, importlib.util, os, shlex, sys

spec = importlib.util.spec_from_file_location('serve', os.path.expanduser('~/.hermes/hy4-single-spark/serve.py'))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

argv = sys.argv[1:]
extra = argv[argv.index('--') + 1:] if '--' in argv else []
argv = argv[:argv.index('--')] if '--' in argv else argv
p = argparse.ArgumentParser()
p.add_argument('--ctx', type=int, default=262144); p.add_argument('--depth', type=int, default=4)
p.add_argument('--ubatch', type=int, default=2048); p.add_argument('--vision', default='gpu')
p.add_argument('--port', type=int, default=8087); p.add_argument('--load-mode', default='dio')
p.add_argument('--cache-ram', type=int, default=0); p.add_argument('--dry-run', action='store_true')
a = p.parse_args(argv)
args = s.materialize(a.ctx, a.depth, a.ubatch, a.vision, a.port, None, a.load_mode, a.cache_ram) + extra
env = os.environ | s.ENV
if a.dry_run:
    print(shlex.join(args)); sys.exit()
os.execve(args[0], args, env)
