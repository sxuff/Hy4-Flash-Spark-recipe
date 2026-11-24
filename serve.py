#!/usr/bin/env python3
"""Hy4 Flash launcher for one GB10 (recipe v2.1, 2026-11-24: v2 plus an MTP draft confidence cutoff). Previous launcher: serve.v1.py."""
import argparse, os, pathlib, subprocess, sys
W=pathlib.Path(__file__).resolve().parent
REV='ecd803c859a36bdf9f3614dbd7c548165df74c30'
MODEL=pathlib.Path.home()/'models'/('Hy4-flash-SPARK-4Bit-Fast-'+REV[:8]); BIN=W/'build-cuda/bin/llama-server'
ENV={'CUDA_VISIBLE_DEVICES':'0','GGML_SCHED_BATCH_HOST_INPUTS':'0','GGML_CUDA_FUSE_HC_GATE':'1','GGML_CUDA_FUSE_HC_CONSUMER':'1','GGML_CUDA_REUSE_Q8':'1','GGML_CUDA_SPARK_MMQ_FUSION':'1','GGML_CUDA_SPARK_MMQ_COMPACT_GRID':'1','GGML_CUDA_SPARK_MMQ_TILE':'64','GGML_CUDA_SPARK_MMQ_TILE_FUSED':'64','GGML_CUDA_MMQ_PREC':'q4','GGML_CUDA_CPY_F32_VEC':'1','GGML_CUDA_SPARK_MMVQ_2W':'1','GGML_CUDA_MOE_CONCURRENT':'1',
     # v2: whole-tile FA for the dense 576/512 MLA layers in prefill (+3-6% prefill, decode unaffected)
     'GGML_CUDA_FA_576_WHOLE_TILES':'1',
     # v2: programmatic dependent launch + graph optimisation (neutral-to-positive on decode)
     'GGML_CUDA_PDL':'1','GGML_CUDA_GRAPH_OPT':'1'}

def materialize(ctx,depth,ubatch,vision,port,override=None,load_mode='dio',cache_ram=0,host='127.0.0.1',p_min=0.75):
  args=[str(BIN),'--model',str(override or MODEL/'Hy4-flash-SPARK-4Bit-Fast.gguf'),'--alias','hy4-flash','--host',host,'--port',str(port),'--ctx-size',str(ctx),'--parallel','1','--n-gpu-layers','999','--split-mode','none',
        # PLE/Engram table (62.7 GiB Q8_0) stays on SSD, read lazily per n-gram; everything else (~87 GiB) is GPU-resident
        '--override-tensor',r'^per_layer_token_embd\.weight$=CPU,^token_embd\.weight$=CUDA0',
        # v2: DirectIO load (~23 s instead of ~320 s); the lazy table is still mmapped
        '--load-mode',load_mode,'--lazy-mode','on','--no-host','--fit','off','--flash-attn','on',
        '--batch-size',str(max(4096,ubatch)),'--ubatch-size',str(ubatch),'--threads','4','--threads-batch','4',
        '--cache-type-k','f16','--cache-type-v','f16','--cache-ram',str(cache_ram),
        '--spec-type','draft-mtp' if depth else 'none','--metrics','--jinja']
  if depth: args+=['--spec-draft-n-max',str(depth)]
  # v2.1: stop drafting once the MTP head is less than p_min sure of the next token (the build's default is 0.00, so
  # every round drafted all `depth` tokens). +7% decode on mixed prose-heavy prompts at temp 0 and 0.7, unchanged on
  # bench/tune.py's code/math/JSON panel at temp 0; drafts only, the target still decides every token
  # (results/skill-hillclimb-20261124/, results/r9-*)
  if depth and p_min: args+=['--spec-draft-p-min',str(p_min)]
  if vision!='off':
   args+=['--mmproj',str(MODEL/'mmproj-Hy4-flash-bf16.gguf')]
   args+=['--mmproj-offload' if vision=='gpu' else '--no-mmproj-offload']
  return args

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--ctx',type=int,default=262144); p.add_argument('--depth',type=int,default=4); p.add_argument('--ubatch',type=int,default=2048)
 p.add_argument('--vision',choices=['gpu','cpu','off'],default='gpu'); p.add_argument('--port',type=int,default=8087)
 p.add_argument('--load-mode',choices=['dio','mmap'],default='dio'); p.add_argument('--cache-ram',type=int,default=0,help='MiB of host RAM prompt cache (0 = off)')
 p.add_argument('--p-min',type=float,default=0.75,help='MTP draft confidence cutoff (0 = draft every round to --depth, the v2 behaviour)')
 p.add_argument('--check',action='store_true'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 assert 1024<=a.port<=65535 and 2048<=a.ctx<=262144 and 0<=a.depth<=8 and 128<=a.ubatch<=4096 and 0<=a.cache_ram<=16384 and 0<=a.p_min<1
 env=os.environ|ENV; args=materialize(a.ctx,a.depth,a.ubatch,a.vision,a.port,W/'intentionally-missing.gguf' if a.check else None,a.load_mode,a.cache_ram,p_min=a.p_min)
 if a.dry_run:
  import shlex; print(' '.join(f'{k}={v}' for k,v in ENV.items()),shlex.join(args)); return
 if a.check:
  r=subprocess.run(args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=25); print(r.stdout)
  assert r.returncode and 'intentionally-missing.gguf' in r.stdout and not ('invalid argument' in r.stdout.lower() or 'unknown argument' in r.stdout.lower()), 'materialized CLI did not reach model inspection'
  return
 os.execve(args[0],args,env)
if __name__=='__main__': main()
