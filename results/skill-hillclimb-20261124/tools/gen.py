#!/usr/bin/env python3
"""Write one bench.py config per arm. Each arm is the recipe v2 default plus ONE change."""
import json, os
H = '~/hillclimb/hy4'
PORT = 19087
ARMS = {  # name: (serve.py args, extra llama-server args, extra env)
    'ctl': ([], [], {}),
    'pmin50': ([], ['--spec-draft-p-min', '0.5'], {}),
    'pmin75': ([], ['--spec-draft-p-min', '0.75'], {}),
    'd5': (['--depth', '5'], [], {}),
    'd6': (['--depth', '6'], [], {}),
    'diag-graphs': ([], [], {'GGML_CUDA_GRAPH_DEBUG': '1'}),
    'pmin90': ([], ['--spec-draft-p-min', '0.9'], {}),
    'd5p75': (['--depth', '5'], ['--spec-draft-p-min', '0.75'], {}),
    'd6p75': (['--depth', '6'], ['--spec-draft-p-min', '0.75'], {}),
    'd8p75': (['--depth', '8'], ['--spec-draft-p-min', '0.75'], {}),
    'd6p90': (['--depth', '6'], ['--spec-draft-p-min', '0.9'], {}),
    'd0': (['--depth', '0'], [], {}),
}

def cfg(name, arm, temperature=0.0, prefill=(4096, 32768)):
    sargs, extra, env = ARMS[arm]
    launch = ['systemd-run', '--user', '--unit', 'hy4-hc', '--collect', '-p', 'MemorySwapMax=0',
              '-p', f'StandardOutput=truncate:{H}/logs/{name}.server.txt', '-p', 'StandardError=inherit']
    launch += [f'--setenv={k}={v}' for k, v in env.items()]
    launch += ['/usr/bin/python3', f'{H}/hc_serve.py', '--port', str(PORT), *sargs]
    if extra:
        launch += ['--', *extra]
    return {
        'name': name, 'base_url': f'http://127.0.0.1:{PORT}', 'model': 'hy4-flash',
        'launch': launch, 'stop': ['systemctl', '--user', 'stop', 'hy4-hc'],
        'health_timeout_s': 600, 'concurrency': 4, 'prefill_lengths': list(prefill),
        'extra_body': {'chat_template_kwargs': {'reasoning_effort': 'no_think'}},
        'temperature': temperature,
        'memory': {'cap_gib': 113, 'probe': ['bash', f'{H}/hostmem.sh'], 'interval_s': 3},
        'require_ac': True,
    }

if __name__ == '__main__':
    import sys
    # usage: gen.py NAME ARM [TEMP]   e.g. gen.py ctl-a ctl   |   gen.py pmin50 pmin50 0.7
    name, arm = sys.argv[1], sys.argv[2]
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    prefill = () if arm.startswith('diag') else (4096, 32768)
    os.makedirs(f'{H}/configs', exist_ok=True); os.makedirs(f'{H}/logs', exist_ok=True)
    json.dump(cfg(name, arm, t, prefill), open(f'{H}/configs/{name}.json', 'w'), indent=1)
    print(name)
