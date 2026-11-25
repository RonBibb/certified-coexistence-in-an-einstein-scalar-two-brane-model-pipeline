"""Tighten the exact critical action and bracket two equilibria at one action."""
import json
from pathlib import Path
import platform
import sys

from flint import arb,ctx,fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum,minimum,require
from coupled_v002 import launch
from fold_identity_controls_v004 import main as exact_controls
from single_coefficient_branch_v003 import root_tube
from single_coefficient_v003 import Cell,Flow


class PointFlow(Flow):
    def __init__(self,A,p,ca,end,steps):
        def rhs(state):
            W,H,phi,T,_,k=state
            return [H,-T*T/3-k*(-2*W).exp(),T,p*phi-4*H*T,(-W).exp(),W*0]
        integrator.rhs=rhs
        y=launch(A,p,ca)[:6]
        self.cells=[]
        h=end/steps
        for n in range(steps):
            z,_=integrator.picard_box(y,h)
            require(all(x.is_finite() for x in z),'finite point trajectory')
            require(minimum(z[0].exp())>fmpq(1,3) and maximum(z[3])<0
                    and minimum(z[5])>0,'positive warp and k, nonzero T')
            cell=Cell(n*h,h,y,z)
            self.cells.append(cell)
            y=cell.evaluate((n+1)*h,(n+1)*h)


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    exact_controls()
    cfg=json.loads(Path('CONFIG_fold_tighten_v004_repair1.json').read_text())
    evidence=json.loads(Path(cfg['fold_evidence']).read_text())
    source=json.loads(Path(cfg['source_enclosures']).read_text())
    ctx.prec=cfg['precision_bits']
    integrator.TAYLOR_ORDER=cfg['taylor_order']
    integrator.MAX_PICARD_ITERATIONS=32
    ca,cb=[integrator.ball(*map(fmpq,source[name])) for name in ('C_a','C_b')]
    lo,hi=map(fmpq,evidence['A'])
    dl,dr=map(fmpq,cfg['D_bracket'])
    p=model.STAR
    pack=integrator.interval_record

    def evaluate(a):
        flow=PointFlow(arb(a),p,ca,dr,cfg['steps'])
        y,D,_=root_tube(flow,p,cb,dl,dr,40)
        return {'A':arb(a),'D':D,'v':y[2]+y[3]/10+(y[2]-cb)/40,'k':y[5]}

    centre=evaluate((lo+hi)/2)
    slopes=list(map(fmpq,evidence['v_A']))
    require(slopes[0]<0<slopes[1],'critical slope enclosure')
    err=max(map(abs,slopes))*(hi-lo)/2
    vf=integrator.ball(minimum(centre['v'])-err,maximum(centre['v']))
    d_slope=max(map(abs,map(fmpq,evidence['D_A'])))
    Df=integrator.ball(minimum(centre['D'])-d_slope*(hi-lo)/2,
                       maximum(centre['D'])+d_slope*(hi-lo)/2)
    v0,v1=map(fmpq,cfg['test_same_action_interval'])
    limit=min(fmpq(evidence[key][0]) for key in ('v_lo','v_hi'))
    require(maximum(vf)<v0<v1<limit,'finite same-action two-equilibrium interval')
    fixed=fmpq(cfg['witness_fixed_v'])
    require(v0<fixed<v1,'interior fixed action')
    a0,a1=map(fmpq,cfg['left_A_bracket'])
    a2,a3=map(fmpq,cfg['right_A_bracket'])
    require(a0<a1<lo<hi<a2<a3,'two disjoint launches outside critical bracket')
    l0,l1,r0,r1=map(evaluate,(a0,a1,a2,a3))
    require(minimum(l0['v'])>fixed>maximum(l1['v']),'same-action left signs')
    require(maximum(r0['v'])<fixed<minimum(r1['v']),'same-action right signs')
    data={'status':'PASS_TIGHT_CRITICAL_ACTION_AND_LOCAL_SAME_ACTION_COEXISTENCE',
          'fold_source':evidence,'fold_v':pack(vf),'fold_D':pack(Df),
          'same_action_open_interval':[str(v0),str(v1)],
          'fixed_witness_v':str(fixed),
          'left_A_bracket':cfg['left_A_bracket'],'right_A_bracket':cfg['right_A_bracket'],
          'four_endpoint_records':[{k:pack(v) for k,v in q.items()} for q in (l0,l1,r0,r1)],
          'scope':'Two equilibria at each fixed action in the interval, stable and one-negative, conditional on the generic radial correspondence and fold sign proof.'}
    Path('FOLD_TIGHTEN_v004_RESULT.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
