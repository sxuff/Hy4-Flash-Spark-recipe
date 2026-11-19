#!/usr/bin/env python3
import fcntl,hashlib,json,os,pathlib,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
G=2**30; M=2**20
REPO='AngelSlim/Hy4-flash-GGUF'; REV='ecd803c859a36bdf9f3614dbd7c548165df74c30'
DEST=pathlib.Path.home()/'models'/('Hy4-flash-SPARK-4Bit-Fast-'+REV[:8]); DEST.mkdir(exist_ok=True)
FILES=(('mmproj-Hy4-flash-bf16.gguf',1044397440,'8c33a48ef57480abf4854126964058a2015b5da3b773b14c847c995cf8899bda'),('Hy4-flash-SPARK-4Bit-Fast.gguf',159611554144,'b402322e67879ce1b65258e2dc1c3b3d6e25eeea074a391b5c29d2ac8b331613'))
FLOOR=20*G
CHUNK=256*M

def emit(name,data):
 f=ROOT/name; t=f.with_suffix('.tmp'); t.write_text(json.dumps(data,indent=2)+'\n'); t.replace(f)
def avail():
 s=os.statvfs(DEST); return s.f_bavail*s.f_frsize
def discard(fd,a,b):
 if b>a: os.posix_fadvise(fd,a,b-a,os.POSIX_FADV_DONTNEED)
def say(*words): print(time.strftime('%FT%TZ',time.gmtime()),*words,flush=True)
def hash_file(f):
 h=hashlib.sha256()
 with f.open('rb',buffering=0) as rd:
  fd=rd.fileno(); p=0
  while b:=rd.read(4*M):
   h.update(b); new=p+len(b)
   if new//(64*M)!=p//(64*M): discard(fd,p//(64*M)*(64*M),new//(64*M)*(64*M))
   p=new
  discard(fd,p//(64*M)*(64*M),p)
 return h

def transfer(name,size,expected,verified):
 final=DEST/name; part=DEST/(name+'.part'); url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
 assert not final.is_symlink() and not part.is_symlink()
 if final.exists():
  assert final.stat().st_size==size and hash_file(final).hexdigest()==expected, f'Existing final file violated seal: {name}'
  return
 if not part.exists(): part.touch(exist_ok=False)
 h=hash_file(part); start=part.stat().st_size
 if not 0<=start<=size: raise ValueError(f'invalid partial length {start}')
 with part.open('r+b',buffering=0) as f:
  fd=f.fileno(); last=0
  while start<size:
   end=min(start+CHUNK,size)-1
   if avail()<FLOOR+end-start+1: raise RuntimeError('20 GiB disk reserve gate')
   for attempt in range(1,7):
    try:
     f.seek(start); f.truncate(start); hc=h.copy(); n=0
     req=urllib.request.Request(url+f'?download=true&offset={start}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'Hy4-GB10-verified-download/1.0'})
     with urllib.request.urlopen(req,timeout=45) as r:
      if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}': raise RuntimeError(f'invalid HTTP range returned: {r.status} {r.headers.get("Content-Range")}')
      while b:=r.read(2*M):
       n+=len(b)
       if n>end-start+1: raise RuntimeError('oversized range reply')
       if f.write(b)!=len(b): raise OSError('short disk write')
       hc.update(b)
      if n!=end-start+1: raise RuntimeError(f'short reply {n}/{end-start+1}')
     os.fdatasync(fd); discard(fd,start,end+1); h=hc; start=end+1; break
    except Exception as e:
     f.seek(start); f.truncate(start); say('range retry',name,start,attempt,type(e).__name__,str(e).split('?')[0]);
     if attempt==6: raise
     time.sleep(min(2**attempt,30))
   if start==size or time.monotonic()-last>12:
    say('downloaded',name,start,size); emit('download-status.json',dict(stage='downloading',name=name,bytes_current=start,bytes_expected=size,verified=verified,updated_at=time.time())); last=time.monotonic()
 got=h.hexdigest()
 if got!=expected: raise RuntimeError(f'LFS digest mismatch for {name}: {got} != {expected}; .part retained')
 part.replace(final)
 with final.open('rb') as f: os.fsync(f.fileno())
 dfd=os.open(DEST,os.O_RDONLY); os.fsync(dfd); os.close(dfd)

def main():
 lock=(ROOT/'download.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=[dict(name=n,bytes=b,sha256=h) for n,b,h in FILES]; emit('download-manifest.json',dict(repo=REPO,revision=REV,path=str(DEST),files=manifest))
 remain=sum(b-((DEST/n).stat().st_size if (DEST/n).exists() else (DEST/(n+'.part')).stat().st_size if (DEST/(n+'.part')).exists() else 0) for n,b,_ in FILES)
 assert avail()>=remain+FLOOR, f'Not enough disk for remaining {remain} + {FLOOR} reserve: {avail()}'
 if '--preflight' in sys.argv: print('ready',remain,avail()); return
 verified=[]
 try:
  for x in manifest:
   transfer(x['name'],x['bytes'],x['sha256'],verified); verified.append(x); say('sha256 verified',x['name'],x['sha256'])
  emit('download-status.json',dict(stage='verified',repo=REPO,revision=REV,path=str(DEST),files=verified,complete=True,updated_at=time.time())); say('download complete')
 except Exception as e:
  emit('download-status.json',dict(stage='failed',repo=REPO,revision=REV,error=str(e).split('?')[0],complete=False,verified=verified,updated_at=time.time())); raise
if __name__=='__main__': main()
