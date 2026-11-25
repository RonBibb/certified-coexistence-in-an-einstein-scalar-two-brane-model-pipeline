"""Independent fixed-point integer background and finite-wall mass challenge."""
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import platform
import sys

from dyadic_interval import BITS, SCALE, I


P = Q('7385903388887613/18014398509481984')
LA = Q('6755388606266369/1125899906842624')
LB = Q('-851862101055633/140737488355328')
V0 = Q('3602879701896397/36028797018963968')


def require(ok,label):
    if not ok:
        raise AssertionError(label)


def low(x):
    return Q(x.lo,SCALE)


def high(x):
    return Q(x.hi,SCALE)


def pack(x):
    return {'exact':x.rational_bounds(),'display':[float(low(x)),float(high(x))]}


def overlaps(x,y):
    return max(x.lo,y.lo)<=min(x.hi,y.hi)


def intersection(x,y):
    return I(max(x.lo,y.lo),min(x.hi,y.hi))


@dataclass
class Series:
    coefficients: list[I]

    def coerce(self,x):
        if isinstance(x,Series):
            return x
        return Series([I.exact(x)]+[I.exact(0)]*(len(self.coefficients)-1))

    def __add__(self,x):
        y=self.coerce(x)
        return Series([a+b for a,b in zip(self.coefficients,y.coefficients,strict=True)])

    __radd__=__add__

    def __neg__(self):
        return Series([-a for a in self.coefficients])

    def __sub__(self,x):
        return self+-self.coerce(x)

    def __rsub__(self,x):
        return self.coerce(x)+-self

    def __mul__(self,x):
        y=self.coerce(x)
        return Series([sum((self.coefficients[j]*y.coefficients[n-j] for j in range(n+1)),I.exact(0))
                       for n in range(len(self.coefficients))])

    __rmul__=__mul__

    def __truediv__(self,n):
        return Series([a/n for a in self.coefficients])

    def square(self):
        return self*self


def curvature(H,phi,T):
    return H.square()-1-T.square()/12+phi.square()*P/12


def launch(A,ca,known_k=None):
    raw=10*(A-V0).square()+LA
    U=raw+(A-ca).square()/4
    H=-U/6
    T=10*(A-V0)+(A-ca)/4
    k=curvature(H,A,T)
    if known_k is not None:
        require(k.contains(known_k),'fixed exact k in dyadic launch')
        k=I.exact(known_k)
    require(k.lo>0,'positive k')
    return [I.exact(1),I.exact(1),H,A,T,k]


def rhs(y,mu=None):
    a,b,H,phi,T,k=y[:6]
    Tp=phi*P-4*H*T
    base=[H*a,-H*b,-T.square()/3-k*b.square(),T,Tp,k*0]
    if mu is None:
        return base
    q,metric,flux=y[6:]
    lam=4*k+mu
    b2=b.square()
    return base+[-Tp*q.square(),a.square()*T.square()*flux,
                 (b2*Q(2,3)-lam*b2.square()*q.square())*metric]


def taylor(y,degree,mu):
    series=[Series([q]+[I.exact(0)]*degree) for q in y]
    for n in range(degree):
        values=rhs(series,mu)
        for a,b in zip(series,values,strict=True):
            a.coefficients[n+1]=b.coefficients[n]/(n+1)
    return [x.coefficients for x in series]


def picard(y,h,mu):
    dt=I.endpoints(0,h)
    z=[a.hull(a+dt*v).pad() for a,v in zip(y,rhs(y,mu),strict=True)]
    for _ in range(96):
        candidate=[a.hull(a+dt*v) for a,v in zip(y,rhs(z,mu),strict=True)]
        if all(a.inside(b) for a,b in zip(candidate,z,strict=True)):
            return z
        z=[a.hull(b).pad() for a,b in zip(candidate,z,strict=True)]
    raise RuntimeError(f'dyadic Picard inclusion failed at {h}')


class Cell:
    def __init__(self,t,h,y,z,order,mu):
        self.t=t
        self.h=h
        self.z=z
        base=taylor(y,order-1,mu)
        rem=taylor(z,order,mu)
        self.c=[a+[b[-1]] for a,b in zip(base,rem,strict=True)]

    def evaluate(self,lo,hi):
        require(self.t<=lo<=hi<=self.t+self.h,'dyadic subtime containment')
        dt=I.endpoints(lo-self.t,hi-self.t)
        out=[]
        for cs,z in zip(self.c,self.z,strict=True):
            q=I.exact(0)
            for c in reversed(cs):
                q=c+dt*q
            out.append(intersection(q,z))
        return out


class Flow:
    def __init__(self,y,end,steps,order,mu=None):
        self.cells=[]
        self.metric_floor=Q(1)
        self.splits=0
        step=end/steps
        for j in range(steps):
            y=self.advance(y,j*step,step,order,mu)

    def advance(self,y,t,h,order,mu,depth=0):
        try:
            z=picard(y,h,mu)
        except RuntimeError:
            if depth==5:
                raise
            self.splits+=1
            mid=self.advance(y,t,h/2,order,mu,depth+1)
            return self.advance(mid,t+h/2,h/2,order,mu,depth+1)
        a,b,H,phi,T,k=z[:6]
        require(low(a)>Q(1,5) and low(b)>0,'positive dyadic warp')
        require((a*b).contains(1),'warp inverse invariant')
        require((curvature(H,phi,T)-k*b.square()).contains(0),'Hamiltonian invariant')
        require(high(H)<0 and high(T)<0 and low(k)>0,'regular monotone field')
        if mu is not None:
            q,metric,_=z[6:]
            require((T*q).contains(1),'metric reciprocal invariant')
            self.metric_floor=min(self.metric_floor,low(metric))
        cell=Cell(t,h,y,z,order,mu)
        self.cells.append(cell)
        return cell.evaluate(t+h,t+h)

    def evaluate(self,lo,hi):
        result=None
        cursor=lo
        for cell in self.cells:
            left=max(lo,cell.t)
            right=min(hi,cell.t+cell.h)
            if left>right:
                continue
            require(left<=cursor,'dyadic complete time coverage')
            y=cell.evaluate(left,right)
            result=y if result is None else [a.hull(b) for a,b in zip(result,y,strict=True)]
            cursor=max(cursor,right)
        require(result is not None and cursor>=hi,'dyadic time interval covered')
        return result


def event(y,cb):
    _,b,H,phi,T,k=y[:6]
    d=phi-cb
    S=T+d/4
    Sp=phi*P-4*H*T+T/4
    g=6*H-LB-d.square()/4-S.square()/10
    G=-2*T*S-6*k*b.square()-S*Sp/5
    return g,G


def refine_event(flow,cb,lo,hi):
    left,_=event(flow.evaluate(lo,lo),cb)
    right,_=event(flow.evaluate(hi,hi),cb)
    _,gradient=event(flow.evaluate(lo,hi),cb)
    require(left.lo>0 and right.hi<0 and gradient.hi<0,'dyadic exact event signs and monotonicity')
    for _ in range(48):
        m=(lo+hi)/2
        f,_=event(flow.evaluate(m,m),cb)
        if f.lo>0:
            lo=m
        elif f.hi<0:
            hi=m
        else:
            err=max(abs(low(f)),abs(high(f)))/(-high(gradient))
            lo=max(lo,m-err)
            hi=min(hi,m+err)
            break
    return lo,hi,[left,right,gradient]


def B(y,mu,side):
    _,b,H,phi,T,_=y[:6]
    pivot=2-b.square()*mu
    require(pivot.lo>0,'dyadic finite oscillator pole separation')
    response=21-pivot.reciprocal()
    return response+2*side*(phi*P/T-4*H)


def eigen_residual(y0,lo,hi,mu,n,order):
    mu=I.exact(mu)
    T=y0[4]
    lam=4*y0[5]+mu
    Ba=B(y0,mu,-1)
    require(Ba.lo>0 and lam.lo>=0,'lower positive metric pencil data')
    q=T.reciprocal()
    F0=-2*lam*q.square()/Ba
    flow=Flow(y0+[q,I.exact(1),F0],hi,n,order,mu)
    y=flow.evaluate(lo,hi)
    Bb=B(y,mu,+1)
    require(Bb.lo>0 and flow.metric_floor>0,'upper positive metric pencil data')
    weight=y[1].square().square()*y[6].square()
    return y[8]-2*lam*weight*y[7]/Bb,flow.metric_floor,flow.splits


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    config=json.loads(Path('CONFIG_dyadic_v003.json').read_text())
    source=json.loads(Path(config['anchor']).read_text())
    nominal=json.loads(Path(config['nominal_reference']).read_text())
    get=lambda key:I.endpoints(*source[key])
    ca,cb,A_reference=map(get,('C_a','C_b','A'))
    n,order=config['steps'],config['order']
    k0=Q(config['k'])
    d0,d1=map(Q,config['D_bracket'])
    # Fresh independent root signs and derivative, prior to comparing any result.
    Alo,Ahi=Q(23,250),Q(47,500)
    big=I.endpoints(Alo,Ahi)
    def k_at(a):
        a=I.exact(a)
        H=-(10*(a-V0).square()+LA+(a-ca).square()/4)/6
        T=10*(a-V0)+(a-ca)/4
        return curvature(H,a,T),H,T
    require(high(k_at(Alo)[0])<k0<low(k_at(Ahi)[0]),'independent lower launch signs')
    _,H,T=k_at(big)
    derivative=-2*H*T/3-T*Q(41,24)+big*P/6
    require(derivative.lo>0,'independent lower launch monotonicity')
    for _ in range(55):
        mid=(Alo+Ahi)/2
        residual=k_at(mid)[0]-k0
        if residual.hi<0:
            Alo=mid
        elif residual.lo>0:
            Ahi=mid
        else:
            err=max(abs(low(residual)),abs(high(residual)))/low(derivative)
            Alo,Ahi=max(Alo,mid-err),min(Ahi,mid+err)
            break
    A=I.endpoints(Alo,Ahi)
    y0=launch(A,ca,k0)
    flow=Flow(y0,d1,n,order)
    lo,hi,event_bounds=refine_event(flow,cb,d0,d1)
    require(hi-lo<Q(config['max_D_width']),'independent D bound width')
    root=flow.evaluate(lo,hi)
    phi,T,H=root[3],root[4],root[2]
    v=phi+T/10+(phi-cb)/40
    U=I.exact(LB)+10*(phi-v).square()+(phi-cb).square()/4
    Up=20*(phi-v)+(phi-cb)/2
    require((H-U/6).contains(0) and (T+Up/2).contains(0),'independent full upper rows')
    require((1-(A-ca)/1000).lo>0 and (1-(phi-cb)/1000).lo>0,'actual S6 phase factors')
    require(overlaps(A,A_reference) and overlaps(v,I.endpoints(*nominal['v1_new']))
            and overlaps(I.endpoints(lo,hi),I.endpoints(*nominal['D'])),
            'cross-backend overlap')
    mu0,mu1=map(Q,config['mass_sign_bracket'])
    left=eigen_residual(y0,lo,hi,mu0,n,order)
    right=eigen_residual(y0,lo,hi,mu1,n,order)
    require(left[0].lo>0 and right[0].hi<0,'independent first mass signs')
    paths=('dyadic_interval.py','dyadic_v003.py','CONFIG_dyadic_v003.json')
    report={'status':'PASS_DYADIC_ARITHMETIC_INDEPENDENT_V003_BACKGROUND_AND_LOW_MASS',
            'bits':BITS,'order':order,'steps':n,
            'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in paths},
            'exact_old_centres':'Common inherited enclosures in '+config['anchor'],
            'A':pack(A),'D':pack(I.endpoints(lo,hi)),'v1_new':pack(v),'endpoint':{name:pack(q) for name,q in
                      zip(('a','a_inverse','H','Phi','T','k'),root,strict=True)},
            'event_endpoint_and_gradient':[pack(q) for q in event_bounds],
            'low_mass_bracket':[str(mu0),str(mu1)],
            'mass_signs':[pack(left[0]),pack(right[0])],
            'metric_floors':[str(left[1]),str(right[1])],
            'picard_subdivisions':[flow.splits,left[2],right[2]],
            'scope':'192-bit integer fixed-point arithmetic, not Arb; shared mathematical first-order theory and inherited exact anchor enclosures'}
    Path('DYADIC_V003_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
