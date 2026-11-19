#!/usr/bin/env python3
"""GX10 private serial qualification. Proceed only after the two sealed prerequisites."""
import base64, contextlib, fcntl, hashlib, json, os, pathlib, re, shlex, signal, statistics, struct, subprocess as sp, sys, threading, time, urllib.request, zlib
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))  # repo root: serve.py, download.py
from serve import W, MODEL, REV, BIN, ENV, materialize
G=2**30; FLOOR=6*G; MAX_HOST_SWAP=G; SYS=['systemctl','--user']; INC='glm53f-gr8.service'; OLD='http://127.0.0.1:8000'; PORT=8097; END=f'http://127.0.0.1:{PORT}'; MARK=W/'restore-required.json'; STAGE=W/'qualification-status.json'
PROMPTS={'code':'Write a detailed Python implementation with comments of a small in-memory LRU cache. Then explain the invariants and write tests.\n\n', 'math':'Explain a proof by induction of the formula for the sum of the first n squares, then give two alternative proofs.\n\n','prose':'Write a long, concrete travel feature about a sleepy coastal town during winter. Use descriptive prose and several scenes.\n\n','structured':'Generate a JSON array of 40 inventory records with distinct IDs, materials, dimensions, and one-sentence notes.\n\n'}

def put(f,d):
 t=f.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n'); t.replace(f)
def phase(s,**kw): put(STAGE,dict(stage=s,updated_at=time.time(),**kw)); print(time.strftime('%FT%TZ',time.gmtime()),s,kw,flush=True)
def run(args,timeout=30,check=True): return sp.run(args,text=True,capture_output=True,timeout=timeout,check=check)
def system_state(u): return run(SYS+['show','-p','ActiveState','--value',u],check=False).stdout.strip()
def http(url,path,payload=None,timeout=8):
 req=urllib.request.Request(url+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 t=time.monotonic()
 with urllib.request.urlopen(req,timeout=timeout) as f: b=f.read()
 return b,time.monotonic()-t

def req(url,path,payload=None,timeout=8):
 b,v=http(url,path,payload,timeout); return json.loads(b),v

def mem():
 d={k:int(v.split()[0])*1024 for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if (k:=l.split(':',1)[0]) in ('MemAvailable','SwapTotal','SwapFree') for v in [l.split(':',1)[1]]}
 return d['MemAvailable'],d['SwapTotal']-d['SwapFree']

def gpu():
 t=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],8).stdout
 return [int(n.strip()) for n in t.splitlines() if n.strip().isdigit()]

def cgroup():
 p=next(l.split(':',2)[-1] for l in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
 return pathlib.Path('/sys/fs/cgroup'+p)

def idle_old():
 assert system_state(INC)=='active'
 ids={x['id'] for x in req(OLD,'/v1/models')[0]['data']}; assert 'glm-5.3-flash' in ids,ids
 text,_=http(OLD,'/metrics'); text=text.decode()
 for metric in ('vllm:num_requests_running','vllm:num_requests_waiting'):
  samples=[float(l.split()[-1]) for l in text.splitlines() if l.startswith(metric+'{')]; assert samples,metric
  if any(samples): return False
 dock=run(['docker','top','glm53f-1x-spark','-eo','pid'],8).stdout; allowed={int(n.strip()) for n in dock.splitlines() if n.strip().isdigit()}; pids=set(gpu())
 return bool(pids) and pids<=allowed

def restore():
 if not MARK.exists(): return {'required':False}
 with (W/'restore.lock').open('a') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX)
  if not MARK.exists(): return {'required':False}
  marker=json.loads(MARK.read_text()); assert marker['incumbent']==INC
  # Evict only clean cache from the experiment's files, not any system cache.
  for f in MODEL.glob('*.gguf'):
   with f.open('rb') as rd: os.posix_fadvise(rd.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  deadline=time.monotonic()+180
  while system_state(INC)!='active' and time.monotonic()<deadline:
   try:
    if mem()[0]>105*G and not gpu(): break
   except Exception: pass
   time.sleep(3)
  if system_state(INC)!='active':
   assert not gpu() and mem()[0]>105*G, 'restore deferred: another compute PID or insufficient free headroom'
   assert not run(['ss','-H','-ltn','sport',f'= :{PORT}'],6).stdout.strip(), 'restore deferred: experiment listener still bound'
   run(SYS+['start',INC],45)
  deadline=time.monotonic()+360
  while time.monotonic()<deadline:
   try:
    health,_=req(OLD,'/health'); ids={x['id'] for x in req(OLD,'/v1/models')[0]['data']}
    if health.get('ready') and 'glm-5.3-flash' in ids: break
   except Exception: pass
   time.sleep(5)
  else: raise RuntimeError('GLM restoration did not reach health and model readiness')
  answer,_=req(OLD,'/v1/chat/completions',{'model':'glm-5.3-flash','messages':[{'role':'user','content':'What is 12 multiplied by 17? Reply with the integer.'}],'temperature':0,'max_tokens':384},100)
  msg=answer['choices'][0]['message']; assert re.search(r'\b204\b',msg.get('content') or ''),'GLM restore canary failed'
  receipt={'required':True,'verified':True,'model':'glm-5.3-flash','health':health,'generation':'204','time':time.time()}; put(W/'restoration.json',receipt); MARK.unlink(); return receipt

def prereqs():
 end=time.monotonic()+12*3600
 phase('waiting-for-download-and-build')
 while time.monotonic()<end:
  try:
   d=json.loads((W/'download-status.json').read_text()); b=json.loads((W/'build-status.json').read_text())
   assert d['complete'] and d['revision']==REV and pathlib.Path(d['path'])==MODEL and b['status']=='complete' and b['upstream']=='2cbca389f515df455412dafa07e2b1359cc72a6d'
   assert sorted((p['name'],p['bytes'],p['sha256']) for p in d['files'])==sorted(__import__('download').FILES)
   for x in d['files']: assert (MODEL/x['name']).stat().st_size==x['bytes']
   assert BIN.is_file(); run(['sha256sum','-c',str(W/'bin.sha256')],25); assert 'GB10' in (W/'devices.txt').read_text()
   return
  except (OSError,ValueError,KeyError,AssertionError): pass
  time.sleep(15)
 raise TimeoutError('Download/build did not produce matching verified prerequisites')

def canary(answer):
 c=answer['choices'][0]; return c.get('finish_reason'),c['message']

def png():
 w=h=192
 def chunk(k,b): return struct.pack('>I',len(b))+k+b+struct.pack('>I',zlib.crc32(k+b))
 row=b'\0'+bytes((239,23,26))*w
 return 'data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(row*h))+chunk(b'IEND',b'')).decode()

class HardGuard(RuntimeError): pass
class Arm:
 def __init__(self, folder, ctx, depth, ubatch, vision, baseline):
  self.p=folder; folder.mkdir(); self.settings=dict(ctx=ctx,depth=depth,ubatch=ubatch,vision=vision); self.args=materialize(ctx,depth,ubatch,vision,PORT); self.baseline=baseline; self.exit=threading.Event(); self.why=None; self.minimum=1<<62; self.peak_swap=0
 def sample(self, telemetry):
  cg=cgroup(); disk=os.statvfs(W); lows=0
  try:
   assert (cg/'memory.swap.max').read_text().strip()=='0'
   while not self.exit.is_set():
    avail,swap=mem(); cgs=int((cg/'memory.swap.current').read_text()); self.minimum=min(avail,self.minimum); self.peak_swap=max(self.peak_swap,swap-self.baseline)
    lows=lows+1 if avail<FLOOR else 0
    row={'time':time.time(),'available':avail,'host_swap_growth':swap-self.baseline,'cgroup_swap':cgs}
    telemetry.write(json.dumps(row)+'\n'); telemetry.flush()
    if lows>=2 or cgs or swap-self.baseline>MAX_HOST_SWAP or avail<3*G:
     self.why=f'safety guard: {row}'; self.stop(kill=True); return
    if os.statvfs(W).f_bavail*disk.f_frsize<15*G:
     self.why='safety guard: disk below 15 GiB'; self.stop(kill=True); return
    self.exit.wait(.5)
  except Exception as e:
   self.why=f'guard unreadable: {e}'; self.stop(kill=True)
 def stop(self,kill=False):
  if self.proc.poll() is None:
   with contextlib.suppress(ProcessLookupError): os.killpg(self.proc.pid,signal.SIGKILL if kill else signal.SIGTERM)
 def finish(self):
  self.stop()
  try: self.proc.wait(15)
  except sp.TimeoutExpired: self.stop(kill=True); self.proc.wait(10)
  self.exit.set(); self.thread.join(4); self.log.close(); self.telemetry.close()
  put(self.p/'lifecycle.json',{'returncode':self.proc.returncode,'minimum_available':self.minimum,'peak_host_swap_growth':self.peak_swap,'guard':self.why})
 def verify(self):
  if self.why: raise HardGuard(self.why)
  if self.proc.poll() is not None: raise RuntimeError(f'llama-server exited {self.proc.returncode}')
 def __enter__(self):
  avail,swap=mem(); assert avail>=12*G and swap-self.baseline<=MAX_HOST_SWAP,(avail,swap,self.baseline)
  put(self.p/'argv.json',{'argv':self.args,'env':ENV,'settings':self.settings}); self.log=(self.p/'server.log').open('w'); self.telemetry=(self.p/'memory.jsonl').open('w')
  self.proc=sp.Popen(self.args,env=os.environ|ENV,stdout=self.log,stderr=sp.STDOUT,start_new_session=True); self.thread=threading.Thread(target=self.sample,args=(self.telemetry,),daemon=True); self.thread.start()
  try:
   deadline=time.monotonic()+420
   while time.monotonic()<deadline:
    self.verify()
    try:
     if req(END,'/health',timeout=2)[0].get('status')=='ok':
      ids=[x['id'] for x in req(END,'/v1/models',timeout=2)[0]['data']]; assert ids==['hy4-flash'],ids
      assert self.proc.pid in gpu(),'CUDA compute PID missing'; self.verify(); return self
    except (OSError,ValueError): pass
    time.sleep(1)
   raise TimeoutError('Hy4 readiness deadline')
  except BaseException:
   self.finish(); raise
 def __exit__(self,*_): self.finish(); self.verify_guard()
 def verify_guard(self):
  if self.why: raise HardGuard(self.why)
 def call(self,name,path,payload,timeout=130):
  self.verify(); out,t=req(END,path,payload,timeout); self.verify(); put(self.p/(name+'.json'),{'request':payload,'response':out,'wall_s':t}); return out,t
 def chat(self,name,content,no_think=True,**extra):
  payload={'model':'hy4-flash','messages':[{'role':'user','content':content}],'max_tokens':384,'temperature':0,'seed':2026,'cache_prompt':False,'stream':False}
  if no_think: payload['chat_template_kwargs']={'reasoning_effort':'no_think'}
  payload.update(extra); return canary(self.call(name,'/v1/chat/completions',payload)[0])
 def smokes(self):
  out={}
  f,x=self.chat('text','Calculate 12 multiplied by 17. Reply with just the integer.',max_tokens=96); out['text']=bool(f=='stop' and re.search(r'\b204\b',x.get('content') or ''))
  try:
   f,x=self.chat('thinking','Use reasoning: what is 17 times 19? Finish with the result.',False,max_tokens=576); out['thinking']=bool(f=='stop' and (x.get('reasoning_content') or x.get('reasoning')) and re.search(r'\b323\b',x.get('content') or ''))
  except Exception as e: out['thinking']=False; out['thinking_error']=str(e)
  if self.settings['vision']!='off':
   try:
    f,x=self.chat('vision',[{'type':'text','text':'Which single color fills this image? Name the color.'},{'type':'image_url','image_url':{'url':png()}}],max_tokens=96); out['vision']=bool(f=='stop' and re.search(r'\bred\b',x.get('content') or '',re.I))
   except Exception as e:
    self.verify(); out['vision']=False; out['vision_error']=str(e)
  tool={'type':'function','function':{'name':'lookup_price','description':'Look up the inventory price for an item.','parameters':{'type':'object','properties':{'sku':{'type':'string'}},'required':['sku']}}}
  try:
   f,x=self.chat('tool','Use lookup_price to get the price for SKU ABC-123. Call the tool.',False,tools=[tool],tool_choice='auto'); calls=x.get('tool_calls') or []; out['tool']=any(c['function']['name']=='lookup_price' and json.loads(c['function']['arguments']).get('sku')=='ABC-123' for c in calls)
  except Exception as e:
   self.verify(); out['tool']=False; out['tool_error']=str(e)
  self.verify(); put(self.p/'smokes.json',out); return out
 def speed(self):
  rows={}
  for k,text in PROMPTS.items():
   out,t=self.call('bench-'+k,'/completion',{'prompt':text,'n_predict':192,'ignore_eos':True,'temperature':0.9,'top_p':1.0,'seed':2026,'cache_prompt':False,'stream':False})
   timing=out['timings']; n=int(timing['predicted_n']); assert n==192 and timing['predicted_per_second']>0,(k,timing)
   rows[k]={'decode_tps':timing['predicted_per_second'],'prefill_tps':timing['prompt_per_second'],'prefill_ms':timing['prompt_ms'],'wall_s':t,'tokens':n,'draft_n':timing.get('draft_n',0),'accepted':timing.get('draft_n_accepted',0)}
  return rows
 def long(self):
  code='RIVER-4862'; unit='The harbor office records tides, small boats, fog, and repairs in a numbered notebook. The note does not contain a passcode.\n'
  x='Remember this access code exactly: '+code+'\n'+unit*1500+'\nWhat was the access code? Reply with the code only.'
  tok=self.call('long-tokenize','/tokenize',{'content':x,'add_special':False},30)[0]['tokens']; assert 15000<len(tok)<self.settings['ctx']-256,len(tok)
  f,msg=self.chat('long-needle',x,max_tokens=128); return {'raw_prompt_tokens':len(tok),'passed':f=='stop' and code in (msg.get('content') or '')}

def write_recipe(dest,final):
 s=final.get('selected'); facts=['# Hy4 Flash on one NVIDIA GB10','','Private qualification. No endpoint is left running.','',f'- Model: `AngelSlim/Hy4-flash-GGUF@{REV}`, `SPARK-4Bit-Fast` and BF16 vision projector.','- Runtime: llama.cpp `2cbca389f515df455412dafa07e2b1359cc72a6d` + AngelSlim Spark patch, CUDA 13.0, SM121a.','- SSD-backed Engram via mmap/lazy and the PLE CPU placement override. No global cache or swap changes.','']
 if s:
  a=s['settings']; facts+=['## Selected qualified operating point','',f"- Median single-stream decode: **{s['median_tps']:.2f} tok/s** over four 192-token workloads.",f"- Smokes: {s['smokes']}",f"- Qualified long prompt: {s.get('long',{})}",'',f"`python3 serve.py --ctx {a['ctx']} --depth {a['depth']} --ubatch {a['ubatch']} --vision {a['vision']}`",'','Only run with exclusive access to the GPU and the same safety reserve. The code binds loopback, never a public interface.']
 else: facts+=['No operating point passed local qualification. See `RESULTS.json` and preserved run logs.']
 facts+=['','The serving recipe is not a production cutover. GLM restoration status is recorded in `restoration.json`.']
 dest.write_text('\n'.join(facts)+'\n')

def execute():
 seen=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()); folder=W/'runs'/seen; folder.mkdir(parents=True)
 summary={'status':'incomplete','run':seen,'arms':[],'selected':None}; work=False
 def one(label,ctx,depth,ubatch=1024,vision='gpu',long=False):
  phase('qualifying',arm=label,run=seen); arm=Arm(folder/label,ctx,depth,ubatch,vision,base_swap)
  entry={'arm':label,'settings':arm.settings,'status':'failed'}
  try:
   with arm as a:
    entry['smokes']=a.smokes(); assert entry['smokes']['text'],'text canary failed'
    entry['rows']=a.speed(); entry['median_tps']=statistics.median(x['decode_tps'] for x in entry['rows'].values()); entry['active_draft']=sum(x['accepted'] for x in entry['rows'].values())
    if depth: assert entry['active_draft']>0,'MTP not active on measured workload'
    if long: entry['long']=a.long()
   entry.update(status='completed',minimum_available=arm.minimum,peak_host_swap_growth=arm.peak_swap)
  except Exception as e: entry['error']=repr(e); print('ARM FAILURE',label,repr(e),flush=True)
  finally: summary['arms'].append(entry); put(folder/'RESULTS.json',summary)
  if arm.why: raise HardGuard(arm.why)
  return entry
 try:
  prereqs()
  for depth in (0,2,4):
   cmd=[sys.executable,str(W/'serve.py'),'--check','--depth',str(depth),'--port',str(PORT)]; r=run(cmd,35,False); (folder/f'parser-{depth}.log').write_text(r.stdout+r.stderr); assert r.returncode==0,'CLI parser preflight failed'
  cg=cgroup(); assert (cg/'memory.swap.max').read_text().strip()=='0'
  phase('waiting-for-verified-idle',run=seen)
  good=0; until=time.monotonic()+2*3600
  while time.monotonic()<until:
   try: good=good+1 if idle_old() else 0
   except Exception: good=0
   if good>=12: break
   time.sleep(5)
  else: raise RuntimeError('No idle GLM window; did not interrupt')
  put(MARK,dict(incumbent=INC,run=seen)); work=True
  run(SYS+['stop',INC],60)
  last=None; good=0
  for _ in range(60):
   current=mem(); unloaded=not gpu() and current[0]>108*G
   good=good+1 if unloaded and last==current[1] else 0; last=current[1]
   if good>=3: break
   time.sleep(2)
  else: raise RuntimeError('GLM did not release GPU/unified memory')
  base_swap=mem()[1]; summary['host_swap_baseline']=base_swap
  baseline=one('32k-serial',32768,0); ctx=32768; ub=1024; vision='gpu'
  if baseline['status']!='completed':
   # Reuse the same artifact; narrow context/workspace to distinguish allocation from architecture.
   ctx=8192; ub=512; baseline=one('8k-serial-recovery',ctx,0,ub)
  if baseline['status']!='completed' or not baseline.get('smokes',{}).get('vision'):
   vision='cpu'; baseline=one('serial-projector-cpu-recovery',ctx,0,ub,vision)
  assert baseline['status']=='completed','No baseline survived bounded compatibility recovery'
  samples=[baseline,one('matched-mtp-2',ctx,2,ub,vision),one('matched-mtp-4',ctx,4,ub,vision)]
  eligible=[a for a in samples if a['status']=='completed']; base_smoke=baseline['smokes']
  eligible=[a for a in eligible if all(not base_smoke.get(k) or a['smokes'].get(k) for k in ('text','vision','tool','thinking'))]
  best=max(eligible,key=lambda a:a['median_tps']); summary['selected']=best; put(folder/'RESULTS.json',summary)
  if ctx>=32768:
   lc=one('64k-selected-long',65536,best['settings']['depth'],ub,vision,long=True)
   if lc['status']=='completed' and lc.get('long',{}).get('passed') and lc['median_tps']>=best['median_tps']*.90 and all(not best['smokes'].get(k) or lc['smokes'].get(k) for k in ('text','vision','tool','thinking')): summary['selected']=lc
  summary['status']='completed'
 except BaseException as e:
  summary['status']='safety_aborted' if isinstance(e,HardGuard) else 'failed'; summary['error']=repr(e)
  print('QUALIFICATION FAILURE',repr(e),flush=True)
 finally:
  if work:
   try: phase('restoring-incumbent',run=seen); summary['restoration']=restore()
   except Exception as e: summary['status']='restoration_failed'; summary['restoration_error']=repr(e)
  put(folder/'RESULTS.json',summary); put(W/'RESULTS.json',summary); write_recipe(W/'RECIPE.md',summary); phase(summary['status'],run=seen)
 return 0 if summary['status']=='completed' else 1

def interrupt(sig,_): raise InterruptedError(f'received signal {sig}')
if __name__=='__main__':
 if sys.argv[1:]==['--restore-only']: print(json.dumps(restore())); sys.exit(0)
 if sys.argv[1:]==['--self-test']:
  assert len(set(PROMPTS))==4 and base64.b64decode(png().split(',',1)[1]).startswith(b'\x89PNG'); assert canary({'choices':[{'finish_reason':'stop','message':{}}]})==('stop',{}); print('CPU launch, replay shape and PNG self-tests OK'); sys.exit(0)
 signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
 with (W/'qualification.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); sys.exit(execute())
