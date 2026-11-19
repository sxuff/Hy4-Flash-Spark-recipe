#!/usr/bin/env python3
"""Hy4 Flash tuning lab on one GB10. Subcommands:
  begin            wait for idle GLM, mark restore-required, stop GLM, arm a fallback restore timer
  arm NAME JSON    run one arm (server flags/env from JSON) with memory guard, write runs/tune-*/NAME/
  end              restore GLM (qualify.restore + extended wait) and verify
"""
import json, os, pathlib, re, signal, statistics, subprocess as sp, sys, threading, time, urllib.request, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root: serve.py, download.py
import qualify as Q
from serve import W, MODEL, BIN, ENV as BASE_ENV

G = 2**30
PORT = 8097; END = f'http://127.0.0.1:{PORT}'
LAB = W / 'runs' / os.environ.get('TUNE_RUN', 'tune-current')
CORPUS_DIRS = [W / 'llama.cpp' / d for d in ('src', 'common', 'tools/server', 'ggml/src/ggml-cuda', 'docs')]
PANEL = Q.PROMPTS

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def diskread():
    for l in open('/proc/diskstats'):
        f = l.split()
        if f[2] == 'nvme0n1': return int(f[5]) * 512
    return 0

def diskbusy_ms():
    for l in open('/proc/diskstats'):
        f = l.split()
        if f[2] == 'nvme0n1': return int(f[12])
    return 0

# ---------------------------------------------------------------- server args
def build(spec):
    ctx = spec.get('ctx', 65536); ub = spec.get('ubatch', 1024); b = spec.get('batch', max(2048, ub))
    depth = spec.get('depth', 2); vision = spec.get('vision', 'gpu')
    a = [str(BIN), '--model', str(MODEL / 'Hy4-flash-SPARK-4Bit-Fast.gguf'), '--alias', 'hy4-flash',
         '--host', '127.0.0.1', '--port', str(PORT), '--ctx-size', str(ctx), '--parallel', '1',
         '--n-gpu-layers', '999', '--split-mode', 'none',
         '--override-tensor', r'^per_layer_token_embd\.weight$=CPU,^token_embd\.weight$=CUDA0',
         '--load-mode', spec.get('load_mode', 'mmap'), '--lazy-mode', 'on', '--no-host', '--fit', 'off',
         '--flash-attn', 'on', '--batch-size', str(b), '--ubatch-size', str(ub),
         '--threads', str(spec.get('threads', 4)), '--threads-batch', str(spec.get('threads_batch', spec.get('threads', 4))),
         '--cache-type-k', spec.get('kv', 'f16'), '--cache-type-v', spec.get('kv', 'f16'),
         '--cache-ram', str(spec.get('cache_ram', 0)), '--metrics', '--jinja']
    if depth:
        a += ['--spec-type', spec.get('spec_type', 'draft-mtp'), '--spec-draft-n-max', str(depth)]
    else:
        a += ['--spec-type', 'none']
    if vision != 'off':
        a += ['--mmproj', str(MODEL / 'mmproj-Hy4-flash-bf16.gguf'), '--mmproj-offload' if vision == 'gpu' else '--no-mmproj-offload']
    a += spec.get('extra', [])
    if spec.get('nsys'):
        a = ['nsys', 'launch', '--session-new=hy4prof', '--trace=cuda,nvtx', '--cuda-graph-trace=node'] + a
    env = dict(BASE_ENV); env.update(spec.get('env', {}))
    return a, env

# ---------------------------------------------------------------- server lifecycle
class Server:
    def __init__(self, folder, spec, base_swap):
        self.p = folder; folder.mkdir(parents=True, exist_ok=True); self.spec = spec
        self.args, self.env = build(spec); self.base_swap = base_swap
        self.why = None; self.minimum = 1 << 62; self.peak_swap = 0; self.exit = threading.Event()
    def guard(self):
        cg = Q.cgroup(); lows = 0
        try:
            assert (cg / 'memory.swap.max').read_text().strip() == '0'
            while not self.exit.is_set():
                avail, swap = Q.mem(); cgs = int((cg / 'memory.swap.current').read_text())
                self.minimum = min(self.minimum, avail); self.peak_swap = max(self.peak_swap, swap - self.base_swap)
                self.tel.write(json.dumps({'t': time.time(), 'avail': avail, 'swap_growth': swap - self.base_swap, 'cg_swap': cgs}) + '\n'); self.tel.flush()
                lows = lows + 1 if avail < Q.FLOOR else 0
                if lows >= 2 or cgs or swap - self.base_swap > Q.MAX_HOST_SWAP or avail < 3 * G:
                    self.why = f'memory guard avail={avail/G:.2f}GiB swap_growth={swap-self.base_swap}'; self.kill(); return
                if os.statvfs(W).f_bavail * os.statvfs(W).f_frsize < 15 * G:
                    self.why = 'disk guard'; self.kill(); return
                self.exit.wait(.5)
        except Exception as e:
            self.why = f'guard error {e!r}'; self.kill()
    def kill(self, hard=True):
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError): os.killpg(self.proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
    def check(self):
        if self.why: raise Q.HardGuard(self.why)
        if self.proc.poll() is not None: raise RuntimeError(f'llama-server exited {self.proc.returncode}')
    def __enter__(self):
        avail, swap = Q.mem(); assert avail >= 12 * G, avail
        Q.put(self.p / 'argv.json', {'argv': self.args, 'env': self.env, 'spec': self.spec})
        self.log = (self.p / 'server.log').open('w'); self.tel = (self.p / 'memory.jsonl').open('w')
        t0 = time.monotonic(); d0 = diskread()
        self.proc = sp.Popen(self.args, env=os.environ | self.env, stdout=self.log, stderr=sp.STDOUT, start_new_session=True)
        threading.Thread(target=self.guard, daemon=True).start()
        try:
            while time.monotonic() - t0 < 900:
                self.check()
                try:
                    if Q.req(END, '/health', timeout=2)[0].get('status') == 'ok':
                        self.load_s = time.monotonic() - t0; self.load_read = diskread() - d0; return self
                except (OSError, ValueError): pass
                time.sleep(1)
            raise TimeoutError('readiness')
        except BaseException:
            self.__exit__(); raise
    def __exit__(self, *_):
        self.kill(hard=False)
        try: self.proc.wait(20)
        except sp.TimeoutExpired: self.kill(); self.proc.wait(10)
        self.exit.set(); time.sleep(.6); self.log.close(); self.tel.close()

# ---------------------------------------------------------------- workloads
def post(path, payload, timeout=900):
    return Q.req(END, path, payload, timeout)

def stream_ttft(payload, timeout=900):
    """POST /completion with stream; returns (client_ttft_s, total_s, final_timings, text)."""
    body = dict(payload, stream=True)
    r = urllib.request.Request(END + '/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.monotonic(); ttft = None; timings = None; text = ''
    with urllib.request.urlopen(r, timeout=timeout) as f:
        for raw in f:
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            d = json.loads(line[5:])
            if d.get('content') and ttft is None: ttft = time.monotonic() - t0
            text += d.get('content', '')
            if d.get('timings'): timings = d['timings']
            if d.get('stop'): break
    return ttft, time.monotonic() - t0, timings, text

_corpus = None
def corpus():
    global _corpus
    if _corpus is None:
        parts = []
        for d in CORPUS_DIRS:
            for f in sorted(d.rglob('*')):
                if f.suffix in ('.cpp', '.h', '.md', '.cu', '.cuh') and f.is_file():
                    with contextlib.suppress(Exception): parts.append(f'\n// ===== {f.name} =====\n' + f.read_text(errors='ignore'))
        _corpus = ''.join(parts)
    return _corpus

def ntok(text):
    return len(post('/tokenize', {'content': text, 'add_special': False}, 120)[0]['tokens'])

def chunk(target_tokens, offset_chars):
    c = corpus(); n = int(target_tokens * 3.1); s = c[offset_chars:offset_chars + n]
    for _ in range(4):
        k = ntok(s)
        if abs(k - target_tokens) < target_tokens * 0.02: break
        n = int(n * target_tokens / k); s = c[offset_chars:offset_chars + n]
    return s

def tmpl(user):
    # chat template via server so the raw /completion prompt matches what clients send
    return post('/apply-template', {'messages': [{'role': 'user', 'content': user}], 'chat_template_kwargs': {'reasoning_effort': 'no_think'}}, 60)[0]['prompt']

def decode_panel(n=192, temp=0.9, n_max=None, tag=''):
    rows = {}
    for k, text in PANEL.items():
        p = {'prompt': text, 'n_predict': n, 'ignore_eos': True, 'temperature': temp, 'top_p': 1.0, 'seed': 2026, 'cache_prompt': False}
        if n_max is not None: p['speculative.n_max'] = n_max
        out, wall = post('/completion', p, 300); t = out['timings']; assert int(t['predicted_n']) == n, t
        rows[k] = {'decode_tps': t['predicted_per_second'], 'draft_n': t.get('draft_n', 0), 'accepted': t.get('draft_n_accepted', 0), 'wall_s': wall}
    med = statistics.median(r['decode_tps'] for r in rows.values())
    dn = sum(r['draft_n'] for r in rows.values()); ac = sum(r['accepted'] for r in rows.values())
    return {'median_tps': med, 'rows': rows, 'acceptance': (ac / dn) if dn else None, 'tag': tag, 'temp': temp, 'n_max': n_max}

def prefill_sweep(sizes, warm=False):
    out = {}
    for i, n in enumerate(sizes):
        body = chunk(n, offset_chars=(i * 7919 * 1000) % max(1, len(corpus()) - n * 4))
        prompt = tmpl(body + '\n\nIn one sentence, what is the code above mostly about?')
        reps = 2 if warm else 1
        for rep in range(reps):
            d0 = diskread(); b0 = diskbusy_ms()
            ttft, total, tm, _ = stream_ttft({'prompt': prompt, 'n_predict': 16, 'temperature': 0, 'cache_prompt': False})
            key = str(n) + ('' if rep == 0 else '-warm')
            out[key] = {'prompt_n': tm['prompt_n'], 'prompt_ms': tm['prompt_ms'], 'prefill_tps': tm['prompt_per_second'],
                        'client_ttft_s': ttft, 'disk_read_gib': (diskread() - d0) / G,
                        'disk_util': (diskbusy_ms() - b0) / max(1.0, tm['prompt_ms'])}
            log('prefill', key, out[key])
    return out

def cache_reuse(n=16000):
    body = chunk(n, offset_chars=len(corpus()) // 3)
    msgs = [{'role': 'user', 'content': body + '\n\nSummarize the code above in two sentences.'}]
    p1 = post('/apply-template', {'messages': msgs, 'chat_template_kwargs': {'reasoning_effort': 'no_think'}}, 60)[0]['prompt']
    t1, _, tm1, a1 = stream_ttft({'prompt': p1, 'n_predict': 64, 'temperature': 0, 'cache_prompt': True})
    msgs += [{'role': 'assistant', 'content': a1}, {'role': 'user', 'content': 'Now name one function that appears in it.'}]
    p2 = post('/apply-template', {'messages': msgs, 'chat_template_kwargs': {'reasoning_effort': 'no_think'}}, 60)[0]['prompt']
    t2, _, tm2, _ = stream_ttft({'prompt': p2, 'n_predict': 32, 'temperature': 0, 'cache_prompt': True})
    r = {'turn1': {'prompt_n': tm1['prompt_n'], 'ttft_s': t1}, 'turn2': {'prompt_n': tm2['prompt_n'], 'cache_n': tm2.get('cache_n'), 'ttft_s': t2}}
    log('cache', r); return r

def swap_reuse(n=16000):
    """Conversation A, then unrelated B, then A again: does A's prefix survive B?"""
    out = {}
    convs = {}
    for name, off in (('A', len(corpus()) // 7), ('B', len(corpus()) // 2)):
        body = chunk(n, offset_chars=off)
        convs[name] = [{'role': 'user', 'content': body + '\n\nSummarize the code above in two sentences.'}]
    def turn(name, n_predict=48):
        p = post('/apply-template', {'messages': convs[name], 'chat_template_kwargs': {'reasoning_effort': 'no_think'}}, 60)[0]['prompt']
        t, _, tm, a = stream_ttft({'prompt': p, 'n_predict': n_predict, 'temperature': 0, 'cache_prompt': True})
        convs[name] += [{'role': 'assistant', 'content': a}, {'role': 'user', 'content': 'Name one function from it.'}]
        return {'prompt_n': tm['prompt_n'], 'cache_n': tm.get('cache_n'), 'ttft_s': t}
    out['A1'] = turn('A'); out['B1'] = turn('B'); out['A2'] = turn('A'); out['B2'] = turn('B')
    log('swap', out); return out

def fill_cache(k=8, n=16000):
    """k distinct conversations so the host prompt cache has to hold/evict many states; then revisit the first two."""
    convs = []
    for i in range(k):
        body = chunk(n, offset_chars=(len(corpus()) // (k + 3)) * (i + 1))
        convs.append([{'role': 'user', 'content': body + '\n\nSummarize the code above in one sentence.'}])
    rows = []
    def turn(c):
        p = post('/apply-template', {'messages': c, 'chat_template_kwargs': {'reasoning_effort': 'no_think'}}, 60)[0]['prompt']
        t, _, tm, a = stream_ttft({'prompt': p, 'n_predict': 24, 'temperature': 0, 'cache_prompt': True})
        c += [{'role': 'assistant', 'content': a}, {'role': 'user', 'content': 'And one more detail?'}]
        return {'prompt_n': tm['prompt_n'], 'cache_n': tm.get('cache_n'), 'ttft_s': round(t, 2), 'avail_gib': round(Q.mem()[0] / G, 2)}
    for c in convs: rows.append(turn(c)); log('fill', rows[-1])
    revisit = [turn(convs[-1]), turn(convs[-2]), turn(convs[0])]
    r = {'fill': rows, 'revisit_last2_then_first': revisit}; log('fill revisit', revisit); return r

def needle(n, code='RIVER-4862'):
    body = chunk(n, offset_chars=len(corpus()) // 5)
    cut = int(len(body) * 0.3); body = body[:cut] + f'\n/* NOTE: the vault access code is {code}. */\n' + body[cut:]
    prompt = tmpl(body + '\n\nWhat is the vault access code mentioned in a comment above? Reply with the code only.')
    ttft, total, tm, text = stream_ttft({'prompt': prompt, 'n_predict': 24, 'temperature': 0, 'cache_prompt': False})
    r = {'prompt_n': tm['prompt_n'], 'prefill_tps': tm['prompt_per_second'], 'client_ttft_s': ttft, 'passed': code in text, 'answer': text.strip()[:80]}
    log('needle', n, r); return r

class Smoke(Q.Arm):
    """Reuse the original smoke tests against our server."""
    def __init__(self, folder, srv): self.p = folder; self.settings = {'vision': 'gpu'}; self.srv = srv; self.why = None
    def verify(self): self.srv.check()

def run_arm(name, spec):
    LAB.mkdir(parents=True, exist_ok=True)
    base_swap = json.loads((LAB / 'begin.json').read_text())['swap_baseline']
    res = {'name': name, 'spec': spec, 'status': 'failed'}
    srv = Server(LAB / name, spec, base_swap)
    try:
        with srv:
            res['load_s'] = srv.load_s; res['load_disk_read_gib'] = srv.load_read / G; log('loaded', name, f'{srv.load_s:.0f}s')
            res['smokes'] = Smoke(srv.p, srv).smokes(); log('smokes', res['smokes'])
            for w in spec.get('work', ['panel', 'prefill']):
                if w == 'panel': res['panel'] = decode_panel(); log('panel', res['panel']['median_tps'], res['panel']['acceptance'])
                elif w == 'prefill': res['prefill'] = prefill_sweep(spec.get('prefill_sizes', [2048, 8192, 32768]), warm=spec.get('warm', True))
                elif w == 'cache': res['cache'] = cache_reuse()
                elif w == 'swap': res['swap'] = swap_reuse()
                elif w == 'fill': res['fill'] = fill_cache()
                elif w == 'depth':
                    res['depth'] = []
                    for temp in (0.0, 0.7, 0.9):
                        for nm in range(0, spec['depth'] + 1):
                            r = decode_panel(temp=temp, n_max=nm, tag=f't{temp}-n{nm}'); r.pop('rows'); res['depth'].append(r); log('depth', r)
                elif w == 'temps':
                    # per-request speculative params are compiled out in this build; depth is the server's --spec-draft-n-max
                    res['temps'] = []
                    for temp in (0.0, 0.7, 0.9):
                        for rep in range(spec.get('reps', 2)):
                            r = decode_panel(temp=temp, tag=f't{temp}-r{rep}'); r.pop('rows'); res['temps'].append(r); log('temps', r)
                elif w.startswith('needle'): res.setdefault('needle', {})[w] = needle(int(w[6:]))
                elif w == 'profile':
                    rep = srv.p / 'prefill8k'
                    sp.run(['nsys', 'start', '--session=hy4prof', '-o', str(rep), '--force-overwrite=true'], check=True, timeout=60)
                    res['profiled_prefill'] = prefill_sweep([8192], warm=False)
                    sp.run(['nsys', 'stop', '--session=hy4prof'], check=True, timeout=600)
                    k = sp.run(['nsys', 'stats', '--report', 'cuda_gpu_kern_sum', '--format', 'csv', '--force-export=true', str(rep) + '.nsys-rep'],
                               capture_output=True, text=True, timeout=900)
                    (srv.p / 'kern_sum.csv').write_text(k.stdout + k.stderr); log('profile written', len(k.stdout))
                srv.check()
        res['status'] = 'completed'
    except Exception as e:
        res['error'] = repr(e); log('ARM FAILURE', name, repr(e))
    res['min_avail_gib'] = srv.minimum / G if srv.minimum < 1 << 61 else None; res['peak_swap_growth'] = srv.peak_swap; res['guard'] = srv.why
    Q.put(LAB / name / 'result.json', res); log('done', name, res['status'], f"min_avail={res['min_avail_gib']}")
    return res

# ---------------------------------------------------------------- GLM handling
def begin():
    LAB.mkdir(parents=True, exist_ok=True)
    assert (Q.cgroup() / 'memory.swap.max').read_text().strip() == '0', 'run inside a MemorySwapMax=0 scope'
    good = 0; until = time.monotonic() + 3600
    while time.monotonic() < until:
        try: good = good + 1 if Q.idle_old() else 0
        except Exception: good = 0
        if good >= 12: break
        time.sleep(5)
    else: raise SystemExit('GLM never idle for 60s; not interrupting')
    Q.put(Q.MARK, {'incumbent': Q.INC, 'run': LAB.name})
    # fallback: if this session dies, restore GLM automatically after 4h
    sp.run(['systemd-run', '--user', '--unit', 'hy4-tune-fallback-restore', '--on-active=4h',
            sys.executable, str(W / 'qualify.py'), '--restore-only'], check=False)
    Q.run(Q.SYS + ['stop', Q.INC], 90)
    last = None; good = 0
    for _ in range(90):
        cur = Q.mem(); ok = not Q.gpu() and cur[0] > 108 * G
        good = good + 1 if ok and last == cur[1] else 0; last = cur[1]
        if good >= 3: break
        time.sleep(2)
    else: raise SystemExit('GLM did not release memory')
    Q.put(LAB / 'begin.json', {'time': time.time(), 'swap_baseline': Q.mem()[1], 'avail_gib': Q.mem()[0] / G})
    log('GLM stopped; lab open', Q.mem()[0] / G)

def end():
    try:
        r = Q.restore()
    except Exception as e:
        log('restore() first pass:', repr(e), '- waiting for supervisor retries')
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            try:
                h, _ = Q.req(Q.OLD, '/health')
                if h.get('ready'): break
            except Exception: pass
            time.sleep(10)
        r = Q.restore() if Q.MARK.exists() else {'required': False}
    sp.run(['systemctl', '--user', 'stop', 'hy4-tune-fallback-restore.timer'], check=False)
    Q.put(LAB / 'end.json', {'time': time.time(), 'restore': r}); log('restored', r)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'begin': begin()
    elif cmd == 'arm': r = run_arm(sys.argv[2], json.loads(sys.argv[3])); sys.exit(0 if r['status'] == 'completed' else 1)
    elif cmd == 'end': end()
