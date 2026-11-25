"""Direct second background sensitivities at the upper-vacuum stability fold."""
import json
from pathlib import Path
import platform
import sys

from flint import arb, ctx, fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum, minimum, require
from coupled_v002 import launch
from single_coefficient_branch_v003 import root_tube
from single_coefficient_v003 import Cell, Flow, event


GAMMA_HALF=fmpq(41,4)


def rhs(p):
    base_rhs=model.make_rhs(p)

    def augmented(y):
        W,H,phi,T,_,k=y[:6]
        W1,H1,phi1,T1,_,k1=y[6:12]
        U,V,W2,H2,phi2,T2,k2=y[18:]
        exp=(-2*W).exp()
        effective=k2-4*k1*W1-2*k*W2+4*k*W1*W1
        U_s=-4*H*U+(k1-2*k*W1)*exp
        V_s=-4*H*V-4*H1*U+effective*exp
        H2_s=-2*(T1*T1+T*T2)/3-effective*exp
        T2_s=p*phi2-4*(H2*T+2*H1*T1+H*T2)
        return base_rhs(y[:18])+[U_s,V_s,H2,H2_s,T2,T2_s,W*0]
    return augmented


class FoldFlow(Flow):
    def __init__(self,A,p,ca,end,steps):
        self.p=p
        integrator.rhs=rhs(p)
        y=launch(A,p,ca)
        T,H,T1,H1=y[3],y[1],y[9],y[7]
        H2=-T1/3
        k2=2*(H1*H1+H*H2)-T1*T1/6+p/6
        y.extend([arb(0),arb(0),arb(0),H2,arb(0),arb(0),k2])
        require(minimum(y[11])>0,'strict K_A')
        self.cells=[]
        self.steps=0
        self.max_splits=0
        step=end/steps
        for i in range(steps):
            y=self.advance(y,i*step,step)

    def advance(self,y,t,h,depth=0):
        try:
            z,_=integrator.picard_box(y,h)
        except RuntimeError:
            if depth==4:
                raise
            self.max_splits+=1
            mid=self.advance(y,t,h/2,depth+1)
            return self.advance(mid,t+h/2,h/2,depth+1)
        W,H,phi,T,_,k=z[:6]
        W1,_,_,_,_,k1=z[6:12]
        require(all(v.is_finite() for v in z),'finite fold trajectory')
        require(minimum(W.exp())>fmpq(1,3),'regular fold warp')
        require(maximum(H)<0 and maximum(T)<0 and minimum(k)>0,'fold monotone positive-k')
        require(model.constraint(z,self.p).contains(0),'fold Hamiltonian')
        require(minimum(k1-2*k*W1)>0,'compensated U nonnegative everywhere')
        require(maximum(W1-k1/(2*k))<0,'normalized metric zero-shot positivity')
        cell=Cell(t,h,y,z)
        self.cells.append(cell)
        self.steps+=1
        return cell.evaluate(t+h,t+h)


def fold_data(y,p,cb):
    W,H,phi,T,_,k=y[:6]
    W1,H1,phi1,T1,_,k1=y[6:12]
    U,V,_,_,phi2,T2,_=y[18:]
    exp=(-2*W).exp()
    d=phi-cb
    S=T+d/4
    Tp=p*phi-4*H*T
    T1s=p*phi1-4*(H1*T+H*T1)
    Hp=-T*T/3-k*exp
    Tpp=p*T-4*(Hp*T+H*Tp)
    hA=T1+GAMMA_HALF*phi1
    hD=Tp+GAMMA_HALF*T
    _,gD=event(y,p,cb)
    require(maximum(gD)<0 and maximum(hD)<0,'transverse folds are upper scalar and metric events')
    gA=6*H1-d*phi1/2-S*(T1+phi1/4)/5
    DA=-gA/gD
    Gd=-k*exp
    Gv=-S/3
    gdA=-(k1-2*k*(W1+H*DA))*exp
    gvA=-(T1+phi1/4+(Tp+T/4)*DA)/3
    hAA=T2+GAMMA_HALF*phi2+(T1s+GAMMA_HALF*T1)*DA
    hDA=T1s+GAMMA_HALF*T1+(Tpp+GAMMA_HALF*Tp)*DA
    Us=-4*H*U+(k1-2*k*W1)*exp
    UA=V+Us*DA
    require(U.overlaps(H1+T*phi1/3),'compensated vs raw overlap')
    N=U-Gd*hA/hD
    Pv=Gv+10*Gd/hD
    require(minimum(Gv)>0 and minimum(Pv)>0,'positive action slope denominators')
    NA=UA-(gdA*hA+Gd*hAA)/hD+Gd*hA*hDA/(hD*hD)
    PvA=gvA+10*gdA/hD-10*Gd*hDA/(hD*hD)
    vA=-N/Pv
    vAA=-(NA*Pv-N*PvA)/(Pv*Pv)
    ratio=Tp/T
    require(minimum(2*GAMMA_HALF+2*ratio)>0,'positive upper shifted wall coefficient')
    return {'v':phi+S/10,'v_A':vA,'v_AA':vAA,'k':k,'D_A':DA,
            'H':H,'T':T,'Phi':phi,'a':W.exp(),'h_D':hD,'G_v':Gv,
            'U':U,'k_A':k1,'phase_c_b':1-d/1000}


def zero_shot(y):
    W,H,_,_,_,k=y[:6]
    W1,_,_,_,_,k1=y[6:12]
    U=y[18]
    a2=(2*W).exp()
    ya=-k1/(2*k)
    psi=W1+ya+H*a2*U/k
    return a2*psi/ya,ya,psi


def rate(flow,endpoints,y,q,p):
    lo,hi=endpoints
    yb,ya,psi=zero_shot(y)
    del psi
    require(minimum(yb)>0,'normalized fold mode at upper wall')
    integral=arb(0)
    for cell in flow.cells:
        left=cell.t
        right=left+cell.h
        if left>hi:
            break
        z=cell.evaluate(left,min(right,hi))
        _,ya_z,psi_z=zero_shot(z)
        weight=(psi_z/(z[3]*ya_z))**2
        # The exact integration length of this cell varies with D in [lo,hi].
        length=integrator.ball(max(fmpq(0),min(right,lo)-left),
                               max(fmpq(0),min(right,hi)-left))
        integral+=length*weight
    a,T,H,phi,k=(q[name] for name in ('a','T','H','Phi','k'))
    Bb=2*GAMMA_HALF+2*(p*phi/T-4*H)
    y0=flow.cells[0].evaluate(fmpq(0),fmpq(0))
    T0,H0,phi0=y0[3],y0[1],y0[2]
    Ba=2*GAMMA_HALF-2*(p*phi0/T0-4*H0)
    require(minimum(Ba)>0 and minimum(Bb)>0,'positive oscillator Robin blocks')
    lam=4*k
    Ma_prime=2/(T0*T0)*(1/Ba+lam/(4*Ba*Ba))
    Mb_prime=2/(a**4*T*T)*(1/Bb+lam/(4*a*a*Bb*Bb))
    Rmu=-(integral+Ma_prime+Mb_prime*yb*yb)/yb
    pref=10-q['h_D']*a*a*q['G_v']/k
    scale=2*pref/(3*T*Bb*ya)
    muA=scale*q['v_AA']/Rmu
    coeff=-muA*(2/q['v_AA']).sqrt()
    require(minimum(scale)>0 and maximum(Rmu)<0 and maximum(muA)<0 and minimum(coeff)>0,
            'strict mass crossing and square-root coefficient')
    return {'metric_integral':integral,'R_mu':Rmu,'mu_A':muA,
            'square_root_C':coeff,'y_b':yb}


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    cfg=json.loads(Path('CONFIG_fold_geometry_v004.json').read_text())
    ctx.prec=cfg['precision_bits']
    integrator.TAYLOR_ORDER=cfg['taylor_order']
    integrator.MAX_PICARD_ITERATIONS=96
    source=json.loads(Path(cfg['source_enclosures']).read_text())
    ca,cb=[integrator.ball(*map(fmpq,source[key])) for key in ('C_a','C_b')]
    p=model.STAR
    a0,a1=map(fmpq,cfg['A_interval'])
    d0,d1=map(fmpq,cfg['D_bracket'])
    pack=integrator.interval_record
    slabs=cfg['A_slabs']
    step=(a1-a0)/slabs

    def evaluate(left,right):
        A=integrator.ball(left,right)
        flow=FoldFlow(A,p,ca,d1,cfg['steps'])
        y,D,_=root_tube(flow,p,cb,d0,d1,cfg['event_refinements'])
        q=fold_data(y,p,cb)
        q.update(A=A,D=D,phase_c_a=1-(A-ca)/1000)
        require(minimum(q['phase_c_a'])>0 and minimum(q['phase_c_b'])>0,
                'positive phase kinetic')
        return q,flow,y

    partial=[]
    for i in range(slabs):
        q,_,_=evaluate(a0+i*step,a0+(i+1)*step)
        partial.append({k:pack(v) for k,v in q.items()})
    Path('FOLD_SLABS_PARTIAL.json').write_text(json.dumps(partial,indent=2)+'\n')
    floors=[fmpq(q['v_AA']['exact_bounds'][0]) for q in partial]
    require(min(floors)>0,'strict geometric convexity across every slab')
    left,_,_=evaluate(a0,a0)
    right,_,_=evaluate(a1,a1)
    require(maximum(left['v_A'])<0 and minimum(right['v_A'])>0,'opposite geometric derivative signs')
    lo,hi=a0,a1
    for _ in range(cfg['fold_refinements']):
        if hi-lo<fmpq(cfg['A_fold_max_width']):
            break
        mid=(lo+hi)/2
        m,_,_=evaluate(mid,mid)
        if maximum(m['v_A'])<0:
            lo=mid
        elif minimum(m['v_A'])>0:
            hi=mid
        else:
            err=max(abs(minimum(m['v_A'])),abs(maximum(m['v_A'])))/min(floors)
            lo,hi=max(lo,mid-err),min(hi,mid+err)
            break
    require(hi-lo<fmpq(cfg['A_fold_max_width']),'unique critical launch width')
    q,flow,y=evaluate(lo,hi)
    require(q['v_A'].contains(0) and minimum(q['v_AA'])>0,'simple minimum enclosure')
    r=rate(flow,(minimum(q['D']),maximum(q['D'])),y,q,p)
    report={'status':'PASS_DIRECT_GEOMETRIC_FOLD_AND_LOCAL_MASS_SCALING',
            'critical':{k:pack(v) for k,v in q.items()},
            'left':{k:pack(v) for k,v in left.items()},
            'right':{k:pack(v) for k,v in right.items()},
            'rate':{k:pack(v) for k,v in r.items()},
            'v_AA_uniform_lower':str(min(floors)),
            'whole_slabs':len(partial),'scope':'Single varying upper scalar vacuum, two original coupled wall oscillators, two phase zeros, complete declared generic radial sector'}
    Path('FOLD_GEOMETRY_v004_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
