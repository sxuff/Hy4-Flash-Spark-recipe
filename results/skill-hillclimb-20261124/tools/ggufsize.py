#!/usr/bin/env python3
"""Sum GGUF tensor bytes by category, from tensor offsets (works for types gguf-py does not know)."""
import struct, sys, collections, re, json
def rd(f, fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
def rstr(f): n = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')
SCAL = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rval(f, t):
    if t in SCAL: return rd(f, SCAL[t])
    if t == 8: return rstr(f)
    if t == 9:
        et = rd(f, 'I'); n = rd(f, 'Q')
        if et in SCAL and n > 64: f.seek(n * struct.calcsize(SCAL[et]), 1); return f'<array {n}>'
        return [rval(f, et) for _ in range(n)]
    raise ValueError(t)
p = sys.argv[1]; f = open(p, 'rb'); import os; size = os.path.getsize(p)
assert f.read(4) == b'GGUF'; ver = rd(f, 'I'); nt = rd(f, 'Q'); nkv = rd(f, 'Q')
meta = {}
for _ in range(nkv):
    k = rstr(f); t = rd(f, 'I'); meta[k] = rval(f, t)
align = meta.get('general.alignment', 32)
tens = []
for _ in range(nt):
    name = rstr(f); nd = rd(f, 'I'); dims = [rd(f, 'Q') for _ in range(nd)]; typ = rd(f, 'I'); off = rd(f, 'Q')
    tens.append([name, dims, typ, off])
data0 = (f.tell() + align - 1) // align * align
tens.sort(key=lambda x: x[3])
for i, t in enumerate(tens):
    end = tens[i + 1][3] if i + 1 < len(tens) else size - data0
    t.append(end - t[3])
cat = collections.Counter(); typ_by = collections.defaultdict(set)
def c(n):
    if n == 'per_layer_token_embd.weight': return 'ple_table_ssd'
    if n == 'token_embd.weight': return 'token_embd'
    if n.startswith('output.'): return 'lm_head'
    if '_exps' in n: return 'routed_experts'
    if '_shexp' in n: return 'shared_experts'
    if re.match(r'blk\.\d+\.(attn|.*indexer)', n): return 'attention'
    if n.startswith('mtp') or '.nextn.' in n: return 'mtp'
    return 'other_dense'
for name, dims, typ, off, nb in tens:
    k = c(name); cat[k] += nb; typ_by[k].add(typ)
arch = meta.get('general.architecture')
keys = {k: v for k, v in meta.items() if arch and k.startswith(arch + '.') and not isinstance(v, list) and not str(v).startswith('<array')}
print(json.dumps({'arch': arch, 'n_tensors': nt, 'meta': keys}, indent=1))
for k, v in cat.most_common(): print(f'{k:16s} {v/2**30:8.2f} GiB  types={sorted(typ_by[k])}')
print('blocks:', len({int(m.group(1)) for n in (t[0] for t in tens) for m in [re.match(r'blk\.(\d+)\.', n)] if m}))
mtp_like = sorted({re.sub(r'\d+', 'N', t[0]) for t in tens if 'nextn' in t[0] or t[0].startswith('mtp')})[:20]; print('mtp tensors:', mtp_like)
