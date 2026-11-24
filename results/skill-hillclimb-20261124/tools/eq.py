#!/usr/bin/env python3
"""Greedy output-equivalence gate (skill, ground rule 6): the skill's bench.py prompt panel at temperature 0,
non-streamed, saved per server, then compared byte for byte.
  eq.py save NAME BASE_URL MODEL [EXTRA_BODY_JSON]   -> logs/eq-NAME.json
  eq.py diff NAME_A NAME_B"""
import hashlib, json, sys, urllib.request
sys.path.insert(0, '~/hillclimb/hy4')
from bench import PROMPTS

def save(name, base, model, extra):
    out = []
    for p in PROMPTS:
        body = {'model': model, 'messages': [{'role': 'user', 'content': p}], 'max_tokens': 512,
                'temperature': 0.0, 'seed': 0} | extra
        req = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        m = json.loads(urllib.request.urlopen(req, timeout=3600).read())['choices'][0]['message']
        t = (m.get('reasoning_content') or '') + '\x00' + (m.get('content') or '')
        out.append(t); print(hashlib.sha256(t.encode()).hexdigest()[:12], len(t), flush=True)
    json.dump(out, open(f'logs/eq-{name}.json', 'w'))

def diff(a, b):
    A, B = json.load(open(f'logs/eq-{a}.json')), json.load(open(f'logs/eq-{b}.json'))
    same = sum(x == y for x, y in zip(A, B))
    for i, (x, y) in enumerate(zip(A, B)):
        if x != y:
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print(f'prompt {i}: first difference at char {k} of {len(x)}/{len(y)}')
    print(f'EQUIVALENCE {a} vs {b}: {same}/{len(A)} identical')

if __name__ == '__main__':
    if sys.argv[1] == 'save':
        save(sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]) if len(sys.argv) > 5 else {})
    else:
        diff(sys.argv[2], sys.argv[3])
