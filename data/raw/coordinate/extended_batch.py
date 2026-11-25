"""Bounded four-core exact-control batch; owned descendants stay in launcher group."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
assert sys.platform=='linux'
cpus=sorted(os.sched_getaffinity(0)); assert len(cpus)==4
import sympy
root=Path(sympy.__file__).resolve().parent
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}
Path('SYMPY_SOURCE_BINDINGS.json').write_text(json.dumps({'root':str(root),'version':sympy.__version__,'files':hashes},indent=2)+'\n')
procs=[];streams=[]
try:
 for mode,cpu in zip(('einstein','scalar','wall','constraints'),cpus):
  out=open('stdout_'+mode,'x');err=open('stderr_'+mode,'x');streams.extend([out,err])
  proc=subprocess.Popen([sys.executable,'-B','extended_worker.py',mode,str(cpu)],stdout=out,stderr=err)
  procs.append((mode,proc))
 Path('BATCH_STARTED.json').write_text(json.dumps({'workers':[{ 'mode':m,'pid':p.pid,'cpu':c} for (m,p),c in zip(procs,cpus)]},indent=2)+'\n')
 results={m:p.wait(timeout=150) for m,p in procs}
 Path('BATCH_END.json').write_text(json.dumps({'exit_codes':results,'all_reaped':all(p.poll() is not None for m,p in procs)},indent=2)+'\n')
 assert all(v==0 for v in results.values())
finally:
 for m,p in procs:
  if p.poll() is None:p.terminate()
 for m,p in procs:
  try:p.wait(timeout=3)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 for f in streams:f.close()
