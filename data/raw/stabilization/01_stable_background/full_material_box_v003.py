"""Joint independent material and background coefficient robustness of v003."""
import argparse
import json
from pathlib import Path
import platform
import sys

from flint import arb,ctx,fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum,minimum,require
from coupled_neighborhood_v002 import contains_strict
from neighborhood_v001 import absolute_upper,ball,matvec,midpoint
from single_coefficient_v003 import long_box


def material(params,wall):
    w,b,e=[params[key+'_'+wall] for key in ('omega2','beta','eta')]
    require(minimum(w)>0 and minimum(b)>0,'positive finite oscillator parameters')
    return w*b*e**2/(w+b)


def launch(A,params,ca,sa):
    p,La,va,ga=[params[key] for key in ('p','LA','v0','gamma_a')]
    H=-(La+ga*(A-va)**2/2+sa*(A-ca)**2/2)/6
    T=(ga*(A-va)+sa*(A-ca))/2
    k=H*H-1-T*T/12+p*A*A/12
    HA=-T/3
    TA=(ga+sa)/2
    kA=2*H*HA-T*TA/6+p*A/6
    return [arb(0),H,A,T,arb(0),k,
            arb(0),HA,arb(1),TA,arb(0),kA,
            arb(0),arb(0),arb(0),arb(0),arb(0),A*A/12]


def flow(A,params,ca,sa,end,steps,tube=None):
    p=params['p']
    model.attach(integrator,p)
    y=launch(A,params,ca,sa)
    require(minimum(A)>0 and minimum(-y[3])>fmpq(3,50),'long-gap lower trace and speed')
    h=end/steps
    for _ in range(steps):
        z,_=integrator.picard_box(y,h)
        long_box(z,p)
        y=integrator.advance(y,z,h)
    if tube:
        y,_=integrator.picard_box(y,tube)
    long_box(y,p)
    return y


def junction(y,params,cb,sb):
    p,Lb,vb,gb=[params[key] for key in ('p','LB','v1','gamma_b')]
    W,H,phi,T,_,k=y[:6]
    _,HA,phiA,TA,_,_=y[6:12]
    U=Lb+gb*(phi-vb)**2/2+sb*(phi-cb)**2/2
    Up=gb*(phi-vb)+sb*(phi-cb)
    G=H-U/6
    h=T+Up/2
    J=[[HA-Up*phiA/6,-k*(-2*W).exp()-T*h/3],
       [TA+(gb+sb)*phiA/2,p*phi-4*H*T+(gb+sb)*T/2]]
    return [G,h],J


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='CONFIG_full_material_box_v003.json')
    cfg=json.loads(Path(parser.parse_args().config).read_text())
    ctx.prec=cfg['precision_bits']
    integrator.TAYLOR_ORDER=cfg['taylor_order']
    anchor=json.loads(Path(cfg['anchor_binding']).read_text())
    nominal=json.loads(Path(cfg['nominal_binding']).read_text())
    exact=lambda name,record: integrator.ball(*map(fmpq,record[name]))
    ca,cb=[exact(name,anchor) for name in ('C_a','C_b')]
    q=lambda key:fmpq(cfg[key])
    x=[q('centre_A'),q('centre_D')]
    rx=[q('radius_A'),q('radius_D')]
    centres={'p':model.STAR,'LA':model.LA,'LB':model.LB,'v0':model.V0,
             'v1':q('centre_v1'),'gamma_a':fmpq(20),'gamma_b':fmpq(20)}
    for wall in ('a','b'):
        for name in ('omega2','beta','eta'):
            centres[name+'_'+wall]=fmpq(1)
    params={k:ball(v,fmpq(cfg['radii'][k])) for k,v in centres.items()}
    A=ball(x[0],rx[0])
    D=ball(x[1],rx[1])
    require(contains_strict(A,exact('A',anchor)) and contains_strict(D,exact('D',nominal))
            and contains_strict(params['v1'],exact('v1_new',nominal)),'nominal v003 lies strictly in box')
    require(fmpq(2,5)<minimum(params['p']) and maximum(params['p'])<fmpq(21,50),'p gap bound')
    require(fmpq(2,5)<minimum(D) and maximum(D)<fmpq(1,2),'D gap bound')
    require(minimum(params['gamma_a'])/2-fmpq(22,5)>5 and
            minimum(params['gamma_b'])/2+fmpq(18,5)-fmpq(21,500)*20>10,'bare trace gap')
    require(min(minimum(params['omega2_'+i]) for i in ('a','b'))*fmpq(11,20)**2>fmpq(1,900),
            'all oscillator kinetic slots have adequate gap')
    sa,sb=material(params,'a'),material(params,'b')
    yc=flow(arb(x[0]),params,ca,sa,x[1],cfg['steps'])
    yx=flow(A,params,ca,sa,x[1]-rx[1],cfg['steps'],2*rx[1])
    c=[1-(A-ca)/1000,1-(yx[2]-cb)/1000]
    require(all(minimum(i)>0 for i in c),'phase kinetic factors')
    F,_=junction(yc,params,cb,sb)
    _,J=junction(yx,params,cb,sb)
    a,b=map(midpoint,J[0])
    c0,d=map(midpoint,J[1])
    det=a*d-b*c0
    require(det!=0,'invertible interval midpoint')
    C=[[d/det,-b/det],[-c0/det,a/det]]
    E=[[arb(int(i==j))-sum(C[i][l]*J[l][j] for l in range(2))
        for j in range(2)] for i in range(2)]
    shifts=matvec(C,F)
    rows=[sum(absolute_upper(v)*r for v,r in zip(row,rx,strict=True))/ri
          for row,ri in zip(E,rx,strict=True)]
    offsets=[absolute_upper(value)/r for value,r in zip(shifts,rx,strict=True)]
    totals=[a+b for a,b in zip(rows,offsets,strict=True)]
    accepted=max(totals)<1
    pack=integrator.interval_record
    report={'status':'PASS_INDEPENDENT_THIRTEEN_COEFFICIENT_STABLE_ACTION_BOX' if accepted else
              'FAILED_STRICT_KRAWCZYK_SELF_MAP',
            'action':'ACTION_v003.txt','parameters':{k:pack(v) for k,v in params.items()},
            'unknown_box':{'A':pack(A),'D':pack(D)},
            'k':pack(yx[5]),'phase_factors':[pack(i) for i in c],
            'relaxed_material_coefficients':[pack(i) for i in (sa,sb)],
            'J':[[pack(v) for v in row] for row in J],
            'row_contraction':[str(v) for v in rows],
            'row_center_offsets':[str(v) for v in offsets],
            'row_total_display':[float(v) for v in totals],
            'full_gap':'4/5625','symmetry_zero_modes':2}
    Path('FULL_MATERIAL_BOX_v003_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    require(accepted,'strict contraction and self-map')


if __name__=='__main__':
    main()
