"""Whole-slab validation of the single varying upper vacuum coefficient."""
import json
from pathlib import Path
import platform
import sys

from flint import arb, ctx, fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum, minimum, require
from coupled_v002 import launch
from single_coefficient_v003 import Cell, Flow, controls, event, long_box


class BranchFlow(Flow):
    def __init__(self,A,p,ca,end,steps):
        rhs=model.make_rhs(p)

        def rhs_with_compensated_sensitivity(y):
            W,H,_,_,_,k=y[:6]
            WA,_,_,_,_,kA=y[6:12]
            U=y[18]
            base=rhs(y[:18])
            return base+[-4*H*U+(kA-2*k*WA)*(-2*W).exp()]

        integrator.rhs=rhs_with_compensated_sensitivity
        y=launch(A,p,ca)+[arb(0)]
        require(minimum(-y[3])>=fmpq(3,50) and minimum(A)>0,'gap launch bound')
        require(minimum(y[11])>0,'strict curvature derivative')
        h=end/steps
        self.cells=[]
        for j in range(steps):
            z,_=integrator.picard_box(y,h)
            long_box(z,p)
            cell=Cell(j*h,h,y,z)
            self.cells.append(cell)
            y=cell.evaluate((j+1)*h,(j+1)*h)


def root_tube(flow,p,cb,lo,hi,refinements):
    gl,_=event(flow.evaluate(lo,lo),p,cb)
    gr,_=event(flow.evaluate(hi,hi),p,cb)
    _,gd=event(flow.evaluate(lo,hi),p,cb)
    require(minimum(gl)>0 and maximum(gr)<0,'uniform opposite g signs')
    require(maximum(gd)<0,'uniform strictly descending g event')
    for _ in range(refinements):
        m=(lo+hi)/2
        g,_=event(flow.evaluate(m,m),p,cb)
        if minimum(g)>0:
            lo=m
        elif maximum(g)<0:
            hi=m
        else:
            err=max(abs(minimum(g)),abs(maximum(g)))/(-maximum(gd))
            lo=max(lo,m-err)
            hi=min(hi,m+err)
            break
    require(lo<hi,'nonempty event tube')
    return flow.evaluate(lo,hi),integrator.ball(lo,hi),(gl,gr,gd)


def quantities(y,p,cb):
    W,H,Phi,T,_,k=y[:6]
    _,HA,PhiA,TA,_,kA=y[6:12]
    U=y[18]
    require(U.overlaps(HA+T*PhiA/3),'compensated derivative crosscheck')
    S=T+(Phi-cb)/4
    v=Phi+S/10
    Ga=U
    Gd=-k*(-2*W).exp()
    Gv=-S/3
    ha=TA+fmpq(41,4)*PhiA
    hd=p*Phi-4*H*T+fmpq(41,4)*T
    require(maximum(hd)<0,'transverse scalar junction')
    num=Ga-Gd*ha/hd
    den=Gv+10*Gd/hd
    require(minimum(num)>0 and minimum(den)>0,'strict decreasing v(A)')
    require(minimum(kA)>0,'strict increasing k(A)')
    return {'v':v,'k':k,'v_A':-num/den,'k_A':kA,'G_A_compensated':U,
            'det_fixed_action':num*hd,'h_D':hd,'a_b':W.exp(),
            'phase_c_b':1-(Phi-cb)/1000}


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    cfg=json.loads(Path('CONFIG_single_coefficient_branch_v003.json').read_text())
    ctx.prec=cfg['precision_bits']
    integrator.TAYLOR_ORDER=cfg['taylor_order']
    source=json.loads(Path(cfg['source_enclosures']).read_text())
    ca,cb=[integrator.ball(*map(fmpq,source[name])) for name in ('C_a','C_b')]
    p=model.STAR
    Al,Ar=map(fmpq,cfg['A_interval'])
    Dl,Dr=map(fmpq,cfg['D_bracket'])
    count=cfg['A_slabs']
    h=(Ar-Al)/count
    pack=integrator.interval_record

    def evaluate(left,right):
        A=integrator.ball(left,right)
        flow=BranchFlow(A,p,ca,Dr,cfg['steps'])
        y,D,signs=root_tube(flow,p,cb,Dl,Dr,cfg['root_bisections'])
        q=quantities(y,p,cb)
        q.update(A=A,D=D,g_left=signs[0],g_right=signs[1],g_D=signs[2],
                 phase_c_a=1-(A-ca)/1000)
        require(minimum(q['phase_c_a'])>0 and minimum(q['phase_c_b'])>0,
                'positive phase inertia')
        return q

    records=[evaluate(Al+i*h,Al+(i+1)*h) for i in range(count)]
    left,right=evaluate(Al,Al),evaluate(Ar,Ar)
    v_lo,v_hi=map(fmpq,cfg['requested_interior_v'])
    require(maximum(right['v'])<v_lo<v_hi<minimum(left['v']), 'certified v interval image')
    require(maximum(right['k'])<fmpq(1,10000), 'gap curvature ceiling')
    require(fmpq(1,900)-fmpq(4,10000)==fmpq(4,5625), 'uniform full gap')
    require(controls(fmpq(1,20000))=='2690/3','gap controls')
    report={'status':'PASS_CONTINUOUS_ONE_COEFFICIENT_STABLE_ACTION_INTERVAL',
            'v_interval':[str(v_lo),str(v_hi)],'full_nonphase_gap':'4/5625',
            'phase_kernel_dimension':2,'exact_source':source,
            'A_left':{k:pack(v) for k,v in left.items()},
            'A_right':{k:pack(v) for k,v in right.items()},
            'slabs':[{k:pack(v) for k,v in q.items()} for q in records]}
    Path('SINGLE_COEFFICIENT_BRANCH_v003_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
