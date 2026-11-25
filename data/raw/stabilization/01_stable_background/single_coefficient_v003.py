"""Validated one-constant upper-wall deformation with original tensions."""
import json
from pathlib import Path
import platform
import sys

from flint import arb, ctx, fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum, minimum, require
from coupled_v002 import intersect, launch


def long_box(y,p,k_min=0,k_max=fmpq(1,10000)):
    W,H,Phi,T,_,k=y[:6]
    a=W.exp()
    require(all(v.is_finite() for v in y),'finite trajectory')
    require(minimum(a)>fmpq(11,20),'long warp bound')
    require(minimum(H)>fmpq(-11,10) and maximum(H)<fmpq(-9,10),'long H bound')
    require(minimum(Phi)>fmpq(-1,10) and maximum(Phi)<fmpq(1,10),'long scalar bound')
    require(minimum(-T)>fmpq(1,20) and maximum(-T)<fmpq(11,20),'long scalar speed')
    require(k_min<minimum(k) and maximum(k)<k_max,'long curvature bound')
    require(model.constraint(y,p).contains(0),'Hamiltonian containment')
    require(minimum(-2*H*(-T)-p*Phi)>0,'monotone shifted weight')
    require(maximum(1-a*a*T*T/(6*k))<0,'long E<0')


class Cell:
    def __init__(self,t,h,y,z):
        self.t=t
        self.h=h
        self.z=z
        order=integrator.TAYLOR_ORDER
        coeff=integrator.taylor_coefficients(y,order-1)
        rem=integrator.taylor_coefficients(z,order)
        self.coeffs=[a+[b[-1]] for a,b in zip(coeff,rem,strict=True)]

    def evaluate(self,left,right):
        require(self.t<=left<=right<=self.t+self.h,'cell subtime containment')
        tau=integrator.ball(left-self.t,right-self.t)
        out=[]
        for z,coeff in zip(self.z,self.coeffs,strict=True):
            value=arb(0)
            for v in reversed(coeff):
                value=v+tau*value
            out.append(intersect(value,z))
        return out


class Flow:
    def __init__(self,A,p,ca,k0,end,steps):
        model.attach(integrator,p)
        y=launch(A,p,ca,k0)
        require(minimum(-y[3])>=fmpq(3,50) and minimum(A)>0,'initial speed/trace for gap')
        h=end/steps
        self.cells=[]
        self.whole=y.copy()
        for j in range(steps):
            z,_=integrator.picard_box(y,h)
            long_box(z,p)
            cell=Cell(j*h,h,y,z)
            self.cells.append(cell)
            self.whole=[integrator.hull(v,w) for v,w in zip(self.whole,z,strict=True)]
            y=cell.evaluate((j+1)*h,(j+1)*h)

    def evaluate(self,left,right):
        out=None
        cursor=left
        for cell in self.cells:
            a=max(left,cell.t)
            b=min(right,cell.t+cell.h)
            if a>b:
                continue
            require(a<=cursor,'coverage without gaps')
            y=cell.evaluate(a,b)
            out=y if out is None else [integrator.hull(v,w) for v,w in zip(out,y,strict=True)]
            cursor=max(cursor,b)
        require(out is not None and cursor>=right,'subtime range covered')
        return out


def event(y,p,cb):
    W,H,Phi,T,_,k=y[:6]
    d=Phi-cb
    S=T+d/4
    Sp=p*Phi-4*H*T+T/4
    g=6*H-model.LB-d*d/4-S*S/10
    gd=-2*T*S-6*k*(-2*W).exp()-S*Sp/5
    return g,gd


def controls(k):
    require(fmpq(5,12)<fmpq(13,20)**2,'endpoint trace first term')
    require(fmpq(1,2)*fmpq(11,60)**2<fmpq(3,20)**2,'endpoint trace derivative term')
    constant=fmpq(2500,9)*(fmpq(12,5)**2*(fmpq(1,5)+fmpq(1,10))+fmpq(3,2))
    require(constant==fmpq(2690,3) and constant<900,'strict shifted gap')
    require(fmpq(1,900)-4*k==fmpq(41,45000),'nominal full gap')
    require(fmpq(1,900)-fmpq(4,10000)==fmpq(4,5625),'uniform full gap')
    return str(constant)


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    config=json.loads(Path('CONFIG_single_coefficient_v003.json').read_text())
    ctx.prec=config['precision_bits']
    integrator.TAYLOR_ORDER=config['taylor_order']
    source=json.loads(Path(config['source_enclosures']).read_text())
    get=lambda key: integrator.ball(*map(fmpq,source[key]))
    ca,cb,A=map(get,('C_a','C_b','A'))
    lo,hi=map(fmpq,config['D_bracket'])
    k0=fmpq(config['k'])
    p=model.STAR
    flow=Flow(A,p,ca,k0,hi,config['steps'])
    ga,_=event(flow.evaluate(lo,lo),p,cb)
    gb,_=event(flow.evaluate(hi,hi),p,cb)
    _,gd=event(flow.evaluate(lo,hi),p,cb)
    require(minimum(ga)>0 and maximum(gb)<0,'opposite strict event endpoint signs')
    require(maximum(gd)<0,'strict monotonicity over entire D bracket')
    for _ in range(config['root_bisections']):
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
    require(hi-lo<fmpq(config['max_event_width']),'tight unique event')
    y=flow.evaluate(lo,hi)
    W,H,Phi,T=y[:4]
    vnew=Phi+T/10+(Phi-cb)/40
    U=arb(model.LB)+10*(Phi-vnew)**2+(Phi-cb)**2/4
    Up=20*(Phi-vnew)+(Phi-cb)/2
    require((H-U/6).contains(0) and (T+Up/2).contains(0),'enclosed upper rows')
    require(minimum(1-(A-ca)/1000)>0 and minimum(1-(Phi-cb)/1000)>0,'positive phase inertia')
    require(not vnew.overlaps(arb(model.V1)), 'only upper vacuum change is nonzero')
    pack=integrator.interval_record
    report={'status':'PASS_SELECTED_COUPLED_ONE_COEFFICIENT_STABILIZATION',
            'input_source':source,'A':pack(A),'D':pack(integrator.ball(lo,hi)),
            'k':str(k0),'v1_new':pack(vnew),'original_v1':str(model.V1),
            'g_left':pack(ga),'g_right':pack(gb),'g_D_uniform':pack(gd),
            'endpoint':{name:pack(value) for name,value in
                        zip(('W','H','Phi','T','Z','k','a','chi','c'),y[:6]+
                            [W.exp(),(Phi-cb)/2,1-(Phi-cb)/1000],strict=True)},
            'whole_path':{name:pack(value) for name,value in
                          zip(('W','H','Phi','T','Z','k'),flow.whole[:6],strict=True)},
            'shifted_inverse_gap_control':controls(k0),'nonphase_gap':'41/45000',
            'phase_kernel_dimension':2}
    Path('SINGLE_COEFFICIENT_v003_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
