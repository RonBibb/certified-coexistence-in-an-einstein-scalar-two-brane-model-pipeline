"""Quantitative open independent seven-coefficient neighborhood for v002."""
import argparse
import json
from pathlib import Path
import platform
import sys

from flint import arb, ctx, fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum, minimum, require
from coupled_v002 import whole_check
from neighborhood_v001 import absolute_upper, ball, matvec, midpoint


def contains_strict(outer,inner):
    return minimum(outer)<minimum(inner) and maximum(inner)<maximum(outer)


def lower_launch(A,params,ca):
    p,La,_,va,_,ga,_ = params
    H=-(La+ga*(A-va)**2/2+(A-ca)**2/4)/6
    T=ga*(A-va)/2+(A-ca)/4
    k=H*H-1-T*T/12+p*A*A/12
    HA=-T/3
    TA=ga/2+fmpq(1,4)
    kA=2*H*HA-T*TA/6+p*A/6
    return [arb(0),H,A,T,arb(0),k,
            arb(0),HA,arb(1),TA,arb(0),kA,
            arb(0),arb(0),arb(0),arb(0),arb(0),A*A/12]


def propagate(A,params,ca,end,steps,tube=None):
    p=params[0]
    model.attach(integrator,p)
    state=lower_launch(A,params,ca)
    step=end/steps
    for _ in range(steps):
        z,_=integrator.picard_box(state,step)
        whole_check(z,p,fmpq(0),fmpq(1,10000))
        state=integrator.advance(state,z,step)
    if tube is not None:
        state,_=integrator.picard_box(state,tube)
    whole_check(state,p,fmpq(0),fmpq(1,10000))
    return state


def upper_quantities(y,params,cb):
    p,_,Lb,_,vb,_,gb=params
    W,H,Phi,T,_,k=y[:6]
    _,HA,PhiA,TA,_,_=y[6:12]
    U=Lb+gb*(Phi-vb)**2/2+(Phi-cb)**2/4
    Up=gb*(Phi-vb)+(Phi-cb)/2
    Upp=gb+fmpq(1,2)
    G=H-U/6
    h=T+Up/2
    J=[[HA-Up*PhiA/6,-k*(-2*W).exp()-T*h/3],
       [TA+Upp*PhiA/2,p*Phi-4*H*T+Upp*T/2]]
    return [G,h],J


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='CONFIG_coupled_neighborhood_v002.json')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    ctx.prec=config['precision_bits']
    integrator.TAYLOR_ORDER=config['taylor_order']
    anchor=json.loads(Path(config['source_enclosures']).read_text())
    inherited=lambda name: integrator.ball(*map(fmpq,anchor[name]))
    ca,cb=inherited('C_a'),inherited('C_b')
    q=lambda name: fmpq(config[name])
    x=[q('centre_A'),q('centre_D')]
    rx=[q('radius_A'),q('radius_D')]
    centres=[model.STAR,model.LA,q('centre_LB'),model.V0,q('centre_v1'),fmpq(20),fmpq(20)]
    names=['p','LA','LB','v0','v1','gamma_a','gamma_b']
    params=[ball(c,fmpq(config['radii'][key])) for key,c in zip(names,centres,strict=True)]
    p,_,Lb,_,vb,ga,gb=params
    A=ball(x[0],rx[0])
    require(contains_strict(A,inherited('A')) and contains_strict(Lb,inherited('LB_new'))
            and contains_strict(vb,inherited('v1_new')), 'exact v002 central action and root inside')
    require(fmpq(2,5)<minimum(p) and maximum(p)<fmpq(21,50), 'bulk restoring interval')
    require(fmpq(1,6)<x[1]-rx[1] and x[1]+rx[1]<fmpq(1,4), 'separation interval')
    require(minimum(ga)>19 and minimum(gb)>19, 'finite positive bare Hessians')
    trace_lb=minimum(ga)/2-fmpq(22,5)
    trace_ub=minimum(gb)/2+fmpq(18,5)-fmpq(21,500)*20
    require(trace_lb>5 and trace_ub>10,'same coercivity traces')
    yc=propagate(arb(x[0]),params,ca,x[1],config['steps'])
    yx=propagate(A,params,ca,x[1]-rx[1],config['steps'],2*rx[1])
    phase_factors=[1-(A-ca)/1000,1-(yx[2]-cb)/1000]
    require(all(minimum(c)>0 for c in phase_factors),'positive phase kinetic factors')
    F0,_=upper_quantities(yc,params,cb)
    _,J=upper_quantities(yx,params,cb)
    a,b=map(midpoint,J[0])
    c,d=map(midpoint,J[1])
    det=a*d-b*c
    require(det!=0,'invertible rational Jacobian')
    C=[[d/det,-b/det],[-c/det,a/det]]
    E=[[arb(int(i==j))-sum(C[i][l]*J[l][j] for l in range(2))
        for j in range(2)] for i in range(2)]
    centre=matvec(C,F0)
    rows=[sum(absolute_upper(e)*r for e,r in zip(row,rx,strict=True))/s
          for row,s in zip(E,rx,strict=True)]
    offsets=[absolute_upper(c)/r for c,r in zip(centre,rx,strict=True)]
    totals=[row+offset for row,offset in zip(rows,offsets,strict=True)]
    pack=integrator.interval_record
    report={'status':'PENDING',
            'action':'ACTION_v002.txt',
            'input_source':anchor,
            'curvature':pack(yx[5]),
            'phase_positive_bounds':[pack(c) for c in phase_factors],
            'J':[[pack(a) for a in row] for row in J],
            'C':[[str(a) for a in row] for row in C],
            'F_center':[pack(a) for a in F0],
            'box':{name:pack(val) for name,val in zip(names,params,strict=True)},
            'unknown_box':{'A':pack(A),'D':[str(x[1]-rx[1]),str(x[1]+rx[1])]},
            'contraction_row':[str(a) for a in rows],
            'offset_row':[str(a) for a in offsets],
            'total_row':[str(a) for a in totals],
            'total_display':[float(a) for a in totals],
            'uniform_full_nonphase_gap':'7/45000'}
    accepted=max(rows)<1 and max(totals)<1
    report['status']='PASS_SEVEN_PARAMETER_COUPLED_STABLE_ACTION_BOX' if accepted else 'FAILED_STRICT_KRAWCZYK_SELF_MAP'
    Path('COUPLED_NEIGHBORHOOD_v002_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    require(accepted,'strict self-map and contraction')


if __name__=='__main__':
    main()
