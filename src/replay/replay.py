#!/usr/bin/env python3
"""Package-relative replay into a new directory. No historical launchers run."""
import argparse, datetime, hashlib, json, os, platform, shutil, signal, subprocess, sys, time
from pathlib import Path

R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['anchor','pair','continuum','fold','integer','robustness','all'],default='all')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cpus',help='Comma-separated Linux CPU IDs already allocated by your host policy')
    ap.add_argument('--timeout',type=int,default=900,help='Wall-clock seconds per stage; failure is preserved, never retried')
    a=ap.parse_args()
    assert sys.platform=='linux' and platform.machine()=='aarch64', 'Original validated scripts require native Linux aarch64.'
    assert not sys.flags.optimize, 'Do not use python -O: scientific predicates are assertions.'
    assert a.timeout>0
    import flint
    assert flint.__version__=='0.9.0' and flint.__FLINT_VERSION__=='3.6.0', 'Use the pinned runtime or qualify another version separately.'
    cpus=list(map(int,a.cpus.split(','))) if a.cpus else [min(os.sched_getaffinity(0))]
    assert 1<=len(cpus)<=4 and set(cpus)<=os.sched_getaffinity(0)
    os.sched_setaffinity(0,set(cpus))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    bindings=read(R/'SOURCE_BINDINGS.json')['sha256']
    assert all(sha(R/p)==h for p,h in bindings.items()), 'Reviewer package identity mismatch'
    sources={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    runtime={'python':platform.python_version(),'machine':platform.machine(),'system':platform.system(),
             'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'python_sha256':sha(Path(sys.executable).resolve())}
    # Bind every package source/extension and loaded shared object, not only __init__.
    package=Path(flint.__file__).parent
    runtime['flint_files']={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.suffix in {'.py','.so'}}
    loaded=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path=line.split()[-1]
        if path.startswith('/') and '.so' in path and Path(path).is_file():loaded.add(Path(path))
    runtime['loaded_libraries']={p.name:sha(p) for p in sorted(loaded)}
    write(out/'RUNTIME.json',runtime);write(out/'INPUTS.json',sources)
    env=os.environ.copy();env.update(STUDY_CPUS=','.join(map(str,cpus)),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    modes=['anchor','pair','continuum','fold','integer'] if a.mode=='all' else (['witness','interval','material'] if a.mode=='robustness' else [a.mode])
    receipt={'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':[],'cpus':cpus,'pass':False,'historical_files_modified':False}
    write(out/'REPLAY.json',receipt)
    try:
        for mode in modes:
            work=out/mode;work.mkdir()
            src=R/'sources'/('constructor' if mode in {'anchor','witness','interval','material'} else 'certifier')
            for p in src.iterdir():
                if p.is_file():shutil.copy2(p,work/p.name)
            for p in (R/'inputs').iterdir():
                if p.is_file():shutil.copy2(p,work/p.name)
            changes={}
            if mode=='anchor':
                shutil.copy2(R/'anchor_check.py',work/'anchor_check.py');cmd=['anchor_check.py'];result='ANCHOR_CHECK.json';flag='pass'
            elif mode=='witness':cmd=['single_coefficient_v003.py'];result='SINGLE_COEFFICIENT_v003_RESULT.json';flag=None
            elif mode=='interval':cmd=['single_coefficient_branch_v003.py'];result='SINGLE_COEFFICIENT_BRANCH_v003_RESULT.json';flag=None
            elif mode=='material':cmd=['full_material_box_v003.py','--config','CONFIG_full_material_box_v003_narrow.json'];result='FULL_MATERIAL_BOX_v003_RESULT.json';flag=None
            elif mode=='pair':cmd=['fixed_v_direct.py','CONFIG_vev_coexistence_v001.json','CONFIG_fixed_v_direct_v001.json'];result='FIXED_V_DIRECT.json';flag='pass'
            elif mode=='continuum':cmd=['vev_coexistence.py','CONFIG_vev_coexistence_v001.json','full'];result='VEV_COEXISTENCE_RESULT.json';flag='certified'
            elif mode=='fold':
                previous=out/'continuum/VEV_COEXISTENCE_RESULT.json'
                data=previous if previous.is_file() else R/'inputs/saved_continuum.json'
                shutil.copy2(data,work/'VEV_COEXISTENCE_RESULT.json')
                cfg=read(work/'CONFIG_vev_fold_finite_v001.json');old=cfg['source_sha256'];cfg['source_sha256']=sha(data)
                write(work/'CONFIG_vev_fold_finite_v001.json',cfg)
                changes={'source_sha256':{'old':old,'new':cfg['source_sha256']},'source':'fresh continuum' if previous.is_file() else 'saved continuum'}
                cmd=['vev_fold_finite.py','CONFIG_vev_coexistence_v001.json','CONFIG_vev_fold_finite_v001.json'];result='VEV_FOLD_FINITE.json';flag='pass'
            else:cmd=['fixed_v_dyadic.py','CONFIG_fixed_v_dyadic_v001.json'];result='FIXED_V_DYADIC.json';flag=None
            write(work/'ADAPTER_CHANGES.json',changes)
            started=time.monotonic();entry={'stage':mode,'command':['python3','-B']+cmd,'timeout':a.timeout}
            with (work/'stdout.txt').open('w') as stdout,(work/'stderr.txt').open('w') as stderr:
                child=subprocess.Popen([sys.executable,'-B']+cmd,cwd=work,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                try:rc=child.wait(timeout=a.timeout)
                except BaseException:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise
            entry.update(exit_code=rc,seconds=time.monotonic()-started,child_reaped=True)
            receipt['stages'].append(entry);write(out/'REPLAY.json',receipt)
            assert rc==0, (mode,'failed; inspect preserved stderr.txt')
            d=read(work/result)
            if flag:assert d[flag]
            else:
                expected={'integer':'PASS_INDEPENDENT_DYADIC_FIXED_ACTION_PAIR_AND_LOWEST_MASSES',
                          'witness':'PASS_SELECTED_COUPLED_ONE_COEFFICIENT_STABILIZATION',
                          'interval':'PASS_CONTINUOUS_ONE_COEFFICIENT_STABLE_ACTION_INTERVAL',
                          'material':'PASS_INDEPENDENT_THIRTEEN_COEFFICIENT_STABLE_ACTION_BOX'}
                assert d['status']==expected[mode]
                if mode=='integer':assert len(d['rows'])==2
            entry['result_sha256']=sha(work/result);entry['pass']=True
            print(json.dumps(entry),flush=True);write(out/'REPLAY.json',receipt)
        receipt['pass']=True
    finally:
        receipt['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt['all_started_children_reaped']=True
        write(out/'REPLAY.json',receipt)
    return 0

if __name__=='__main__':sys.exit(main())
