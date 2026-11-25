"""Native separate state chart and exact record audit for peer v003."""
import hashlib,json,os,sys
from pathlib import Path
from fractions import Fraction
from flint import arb,fmpq,ctx
import ivp_engine as E
assert sys.platform=="linux";ctx.prec=768;ctx.threads=1
C=json.loads(Path(sys.argv[1]).read_text());L=json.loads(Path(C["control_literals"]).read_text())
q=lambda s:fmpq(str(Fraction(str(s))))
p,v0,v1,la,lb=[q(L[z]) for z in ("original_p","v0","v1","lambda_a","lambda_b")]
lo=lambda a:E.bounds(a)[0];hi=lambda a:E.bounds(a)[1]
def load(key):
    r=Path(C[key+"_native"]).read_bytes();assert hashlib.sha256(r).hexdigest()==C[key+"_sha256"];return json.loads(r)
S,R=load("candidate"),load("branch")
assert S["status"]=="PASS_SELECTED_COUPLED_ONE_COEFFICIENT_STABILIZATION"
assert R["status"]=="PASS_CONTINUOUS_ONE_COEFFICIENT_STABLE_ACTION_INTERVAL"
def ex(x):
    a,b=map(q,x["exact_bounds"] if isinstance(x,dict) else x);assert a<=b;return a,b
def box(x):return E.ball(*ex(x))
def pack(x):
    a,b=E.bounds(x);return {"exact":[str(a),str(b)],"display":[float(a),float(b)]}
def cap(a,b):
    l,u=max(lo(a),lo(b)),min(hi(a),hi(b));assert l<=u;return E.ball(l,u)
def atomic(name,x):
    t=Path(name+".tmp")
    with t.open("w") as f:f.write(json.dumps(x,indent=2)+"\n");f.flush();os.fsync(f.fileno())
    t.replace(name)
ca,cb=map(box,[S["input_source"][x] for x in ("C_a","C_b")])
k=q(S["k"]);assert k==q("1/20000")
def lower(A):
    da,dc=A-v0,A-ca
    H=-(la+10*da*da+dc*dc/4)/6;T=10*da+dc/4
    K=H*H-1-T*T/12+p*A*A/12
    Ka=-2*H*T/3-T*q("41/4")/6+p*A/6
    return H,T,K,Ka
def root():
    a,b=q(".092"),q(".094");A=E.ball(a,b)
    assert hi(lower(arb(a))[2])<k<lo(lower(arb(b))[2]) and lo(lower(A)[3])>0
    for j in range(12):
        c=(lo(A)+hi(A))/2;A=cap(A,arb(c)-(lower(arb(c))[2]-k)/lower(A)[3])
    A=cap(A,box(S["A"]));H,T,K,_=lower(A)
    assert K.contains(arb(k)) and hi(T)<-q(".06") and lo(A)>0
    return [arb(0),H,A,T]
def event(y):
    W,H,P,T=y[:4];z=P-cb;X=T+z/4;tp=p*P-4*H*T
    return 6*H-lb-z*z/4-X*X/10,-2*T*X-6*k*(-2*W).exp()-X*(tp+T/4)/5
def rhs(mu):
    def f(z):
        W,H,P,T=z[:4];em=(-2*W).exp();tp=p*P-4*H*T
        a=[H,-T*T/3-k*em,T,tp]
        if mu is not None:
            r,y,j=z[4:];lam=mu+4*k
            a.extend([-tp*r*r, (2*W).exp()*T*T*j, ((2*em/3)-lam*em*em*r*r)*y])
        return a
    return f
def wall(W,H,P,T,r,mu,lower):
    em=(-2*W).exp()
    assert lo(2-mu*em)>0
    G=21-1/(2-mu*em)
    slope=(p*P-4*H*T)*r
    B=G+(-2 if lower else 2)*slope
    assert lo(B)>0
    return 2*em*em*r*r/B
class Cell:
    def __init__(self,s,h,y,z):
        self.s,self.h,self.z=s,h,z
        n=E.TAYLOR_ORDER
        self.c=[x+[b[n]] for x,b in zip(E.taylor_coefficients(y,n-1),E.taylor_coefficients(z,n))]
    def eval(self,l,u):
        assert self.s<=l<=u<=self.s+self.h
        t=E.ball(l-self.s,u-self.s);out=[]
        for c,z in zip(self.c,self.z):
            v=arb(0)
            for x in reversed(c):v=x+t*v
            out.append(cap(v,z))
        return out
class Flow:
    def __init__(self,launch,end,mu=None):
        y=launch.copy();self.h=end/C["steps"];self.cells=[];self.min_y=q(2)
        if mu is not None:
            W,H,P,T=y;r=1/T
            y.extend([r,arb(1),-(mu+4*k)*wall(W,H,P,T,r,mu,True)])
        E.rhs=rhs(mu)
        self.min_a=q(1);self.max_pic=0
        for j in range(C["steps"]):
            z,pic=E.picard_box(y,self.h);self.max_pic=max(pic,self.max_pic)
            W,H,P,T=z[:4];em=(-2*W).exp()
            assert all(v.is_finite() for v in z)
            assert (H*H-1-T*T/12+p*P*P/12-k*em).contains(0)
            assert lo(W.exp())>q(".55") and lo(H)>-q("1.1") and hi(H)<-q(".9")
            assert lo(P)>-q(".1") and hi(P)<q(".1") and lo(T)>-q(".55") and hi(T)<-q(".05")
            assert lo(-2*H*(-T)-p*P)>0
            if mu is not None:
                assert (T*z[4]-1).contains(0)
                self.min_y=min(self.min_y,lo(z[5]))
            self.min_a=min(self.min_a,lo(W.exp()))
            c=Cell(j*self.h,self.h,y,z);self.cells.append(c);y=c.eval((j+1)*self.h,(j+1)*self.h)
    def eval(self,l,u):
        cursor=l;out=None
        for c in self.cells:
            a,b=max(l,c.s),min(u,c.s+c.h)
            if a>b:continue
            assert a<=cursor;cursor=max(cursor,b)
            v=c.eval(a,b);out=v if out is None else [E.hull(x,y) for x,y in zip(out,v)]
        assert cursor==u and out is not None
        return out

def branch_audit():
    assert R["exact_source"]["sha256"]==S["input_source"]["sha256"]
    assert len(R["slabs"])==12 and R["phase_kernel_dimension"]==2
    vL,vH=map(q,R["v_interval"]);assert vL<vH
    assert ex(R["A_left"]["v"])[0]>vH and ex(R["A_right"]["v"])[1]<vL
    prev=None
    for r in R["slabs"]:
        A,D,K=map(ex,[r["A"],r["D"],r["k"]])
        if prev is not None:assert prev[0]<A[0]<=prev[1]<A[1]
        prev=A
        assert q("2/5")<=D[0]<D[1]<=q("1/2")
        assert 0<K[0]<=K[1]<=q("1/10000")
        for name in ["g_left","k_A","G_A_compensated","phase_c_a","phase_c_b","a_b"]:
            assert ex(r[name])[0]>0
        for name in ["g_right","g_D","v_A","det_fixed_action","h_D"]:
            assert ex(r[name])[1]<0
    assert q(R["full_nonphase_gap"])==q("4/5625")
    return {"pass":True,"slabs":12,"v_interval":R["v_interval"],"uniform_gap":R["full_nonphase_gap"],"scope":"Exact interval-record checks, not a second continuum flow."}

def main():
    y0=root();flow=Flow(y0,q(".5"))
    a,b=q(".4"),q(".5")
    f0,_=event(flow.eval(a,a));f1,_=event(flow.eval(b,b));_,der=event(flow.eval(a,b))
    # The derivative on a single giant state hull can be over-wide.
    # Bound each proper-time cell instead, retaining the chronological cover.
    ds=[]
    for c in flow.cells:
        if c.s+c.h<a:continue
        gd=event(c.z)[1];assert hi(gd)<0
        ds.append((lo(gd),hi(gd)))
    assert lo(f0)>0>hi(f1) and ds
    slope=E.ball(min(x for x,y in ds),max(y for x,y in ds))
    for _ in range(C["max_bisections"]):
        if b-a<=q(C["event_width"]):break
        t=(a+b)/2;g=event(flow.eval(t,t))[0]
        if lo(g)>0:a=t
        elif hi(g)<0:b=t
        else:
            n=arb(t)-g/slope
            a,b=max(a,lo(n)),min(b,hi(n));break
    assert a<b and b-a<=q(C["event_width"])
    d=cap(E.ball(a,b),box(S["D"]));dl,dh=E.bounds(d)
    W,H,P,T=flow.eval(dl,dh);X=T+(P-cb)/4
    v=P+X/10
    U=lb+10*(P-v)*(P-v)+(P-cb)*(P-cb)/4
    Up=20*(P-v)+(P-cb)/2
    assert (H-U/6).contains(0) and (T+Up/2).contains(0)
    assert cap(v,box(S["v1_new"])).is_finite() and not v.overlaps(arb(v1))
    phases=[1-(y0[2]-ca)/1000,1-(P-cb)/1000];assert min(map(lo,phases))>0
    # Coercivity product identity and sharper independent lower-bound arithmetic.
    coef=q("2500/9")*(q("12/5")**2*(q("1/5")+q("1/10"))+q("3/2"))
    assert coef==q("2690/3") and coef<900 and 1/q(900)-4*k==q(S["nonphase_gap"])
    background={"pass":True,"launch":pack(y0[2]),"D":pack(d),"g_left":pack(f0),"g_right":pack(f1),
                "g_D":pack(slope),"upper_v":pack(v),"phase_c":[pack(x) for x in phases],
                "gap":S["nonphase_gap"],"branch_audit":branch_audit()}
    atomic("V003_BACKGROUND.json",background)
    def residual(mu):
        f=Flow(y0,dh,mu);W,H,P,T,r,y,j=f.eval(dl,dh)
        return j-(mu+4*k)*wall(W,H,P,T,r,mu,False)*y,f.min_y
    a,b=map(q,C["mass_bracket"]);l,_=residual(arb(a));r,_=residual(arb(b))
    assert lo(l)>0>hi(r),("strict initial mass signs",pack(l),pack(r))
    hist=[{"m":str(a),"g":pack(l)},{"m":str(b),"g":pack(r)}]
    for _ in range(C["max_bisections"]):
        if b-a<=q(C["mass_width"]):break
        t=(a+b)/2;z,_=residual(arb(t));hist.append({"m":str(t),"g":pack(z)})
        if lo(z)>0:a=t
        elif hi(z)<0:b=t
        else:break
    atomic("V003_MASS_HISTORY.json",hist)
    assert b-a<=q(C["mass_width"])
    _,ym=residual(E.ball(a,b));assert ym>0
    out={"pass":True,"background":background,"min_positive_mass":pack(E.ball(a,b)),"metric_y_floor_exact":str(ym),
         "lowest_proof":"Positive y whole bracket and strict sign change, positive spectral weights monotone in mu, global full-form coercivity.",
         "cpus":sorted(os.sched_getaffinity(0))}
    atomic("V003_CHALLENGE.json",out)
    print(json.dumps({"pass":True,"D":background["D"]["display"],"v1_new":background["upper_v"]["display"],"ground_mass":out["min_positive_mass"]["display"],"finite_action_interval":R["v_interval"]}))
if __name__=="__main__":main()
