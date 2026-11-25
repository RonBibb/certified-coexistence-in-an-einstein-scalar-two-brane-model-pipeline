"""Validate the lowest v003 nonsymmetry mass in the finite metric Sturm chart."""
import json
from pathlib import Path
import platform
import sys

from flint import arb,ctx,fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum,minimum,require
from coupled_v002 import launch
from single_coefficient_v003 import long_box


def trace_B(y,p,mu,side):
    W,H,Phi,T=y[:4]
    inva2=(-2*W).exp()
    denom=2-mu*inva2
    require(minimum(denom)>0,'away from true oscillator poles')
    Gamma=21-1/denom
    ratio=p*Phi/T-4*H
    B=Gamma+2*side*ratio
    require(minimum(B)>0,'positive endpoint atom')
    return B


def reciprocal(value):
    if not isinstance(value,integrator.Series):
        require(maximum(value)<0,'strict nonzero T for metric chart')
        return 1/value
    source=value.coefficients
    require(maximum(source[0])<0,'strict nonzero T for Taylor chart')
    coeff=[1/source[0]]
    for degree in range(1,len(source)):
        term=sum(source[index]*coeff[degree-index] for index in range(1,degree+1))
        coeff.append(-term/source[0])
    return integrator.Series(coeff)


def mass_flow(A,ca,p,k0,Dl,Dr,mu,n):
    lam=mu+4*k0
    require(minimum(lam)>0,'strict positive shift')
    y=launch(A,p,ca,k0)[:6]
    qa=1/y[3]
    Ba=trace_B(y,p,mu,-1)
    F0=-2*lam*qa**2/Ba
    y.extend((arb(1),F0))
    def combined(z):
        W,H,Phi,T,_,k,metric,flux=z
        q=reciprocal(T)
        Tp=p*Phi-4*H*T
        base=[H,-T*T/3-k*(-2*W).exp(),T,Tp,(-W).exp(),W*0]
        restoring=fmpq(2,3)*(-2*W).exp()-lam*(-4*W).exp()*q*q
        return base+[(2*W).exp()*T*T*flux,restoring*metric]

    integrator.rhs=combined
    h=Dl/n
    metric_min=fmpq(1)
    splits=0

    def advance(state,step,depth=0,endpoint_tube=False):
        nonlocal metric_min,splits
        try:
            box,_=integrator.picard_box(state,step)
        except RuntimeError as error:
            if depth==5 or endpoint_tube:
                raise RuntimeError(f'Picard subdivision limit at mass {mu}, step {step}') from error
            splits+=1
            middle=advance(state,step/2,depth+1)
            return advance(middle,step/2,depth+1)
        long_box(box,p)
        metric_min=min(metric_min,minimum(box[6]))
        return box if endpoint_tube else integrator.advance(state,box,step)

    for _ in range(n):
        y=advance(y,h)
    if Dl<Dr:
        y=advance(y,Dr-Dl,endpoint_tube=True)
    metric_min=min(metric_min,minimum(y[6]))
    require(metric_min>0,'positive y on whole path')
    Bb=trace_B(y,p,mu,+1)
    weight_b=(-4*y[0]).exp()/y[3]**2
    residual=y[7]-2*lam*weight_b*y[6]/Bb
    return residual,metric_min,{'Ba':Ba,'Bb':Bb,'y_b':y[6],'F_b':y[7]},splits


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    config=json.loads(Path('CONFIG_lowest_mass_v003.json').read_text())
    ctx.prec=config['precision_bits']
    integrator.TAYLOR_ORDER=config['taylor_order']
    integrator.MAX_PICARD_ITERATIONS=96
    data=json.loads(Path(config['anchor_binding']).read_text())
    nominal=json.loads(Path(config['nominal_binding']).read_text())
    A,ca=[integrator.ball(*map(fmpq,data[key])) for key in ('A','C_a')]
    Dl,Dr=map(fmpq,nominal['D'])
    k0=fmpq(config['k'])
    p=model.STAR
    lo,hi=map(fmpq,config['initial_mu'])
    require(0<lo<hi<fmpq(1,10),'low-mass band')
    pack=integrator.interval_record
    history=[]

    def point(mu):
        res,m,_,splits=mass_flow(A,ca,p,k0,Dl,Dr,arb(mu),config['steps'])
        history.append({'mu':str(mu),'residual':pack(res),'minimum_y':str(m),'splits':splits})
        Path('MASS_POINT_HISTORY.json').write_text(json.dumps(history,indent=2)+'\n')
        return res

    flo,fhi=point(lo),point(hi)
    require(minimum(flo)>0 and maximum(fhi)<0,'strict initial residual signs')
    for _ in range(config['bisections']):
        m=(lo+hi)/2
        f=point(m)
        if minimum(f)>0:
            lo,flo=m,f
        elif maximum(f)<0:
            hi,fhi=m,f
        else:
            break
    require(hi-lo<fmpq(config['max_mass_width']),'strict final bracket width')
    res,m,tube,splits=mass_flow(A,ca,p,k0,Dl,Dr,integrator.ball(lo,hi),config['steps'])
    require(res.contains(0),'uniform mass tube includes matching')
    report={'status':'PASS_LOWEST_NONSYMMETRY_MASS_BRACKET',
            'mu_exact_bracket':[str(lo),str(hi)],'mu_decimal_bracket':[float(lo),float(hi)],
            'left_residual':pack(flo),'right_residual':pack(fhi),
            'history':history,'whole_mass_tube_y_minimum':str(m),
            'whole_mass_tube_y_minimum_display':float(m),'whole_mass_tube_picard_splits':splits,
            'whole_mass_tube_quantities':{key:pack(v) for key,v in tube.items()},
            'action_nominal':nominal,'anchor_binding':data,
            'method':'Metric finite-wall Sturm chart with independent oscillators retained in both exact low-band wall rows'}
    Path('LOWEST_MASS_v003_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
