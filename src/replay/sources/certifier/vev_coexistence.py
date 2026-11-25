"""Whole-A outward one-coefficient coexistence, metric sign gate and fold tangent."""
import concurrent.futures as CF,datetime,hashlib,json,multiprocessing as MP,os,sys,time,traceback
from pathlib import Path
from fractions import Fraction
from flint import arb,fmpq,ctx
import ivp_engine as E
assert sys.platform=="linux";ctx.prec=768;ctx.threads=1
C=json.loads(Path(sys.argv[1]).read_text());L=json.loads(Path("CONFIG_controls_v001.json").read_text())
S0=json.loads(Path(C["candidate_config"]).read_text())
raw=Path(S0["candidate_native"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==S0["candidate_sha256"]
SOURCE=json.loads(raw);assert SOURCE["status"]=="PASS_SELECTED_COUPLED_ONE_COEFFICIENT_STABILIZATION"
q=lambda x:fmpq(str(Fraction(str(x))))
p,v0,la,lb=[q(L[n]) for n in ("original_p","v0","lambda_a","lambda_b")]
ca,cb=[E.ball(*map(q,SOURCE["input_source"][n])) for n in ("C_a","C_b")]
lo=lambda a:E.bounds(a)[0];hi=lambda a:E.bounds(a)[1]
def pack(x):
    l,h=E.bounds(x);return {"exact":[str(l),str(h)],"display":[float(l),float(h)]}
def cap(a,b):
    l,h=max(lo(a),lo(b)),min(hi(a),hi(b));assert l<=h;return E.ball(l,h)
def atomic(name,x):
    t=Path(name+".tmp")
    with t.open("w") as f:json.dump(x,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    t.replace(name)
def inverse(x):
    if not isinstance(x,E.Series):
        assert hi(x)<0 or lo(x)>0
        return 1/x
    a=x.coefficients;u=1/a[0];v=[u]
    for n in range(1,len(a)):
        v.append(-u*sum((a[j]*v[n-j] for j in range(1,n+1)),arb(0)))
    return E.Series(v)
def base(A):
    d=A-v0;e=A-ca
    H=-(la+10*d*d+e*e/4)/6;T=10*d+e/4
    K=H*H-1-T*T/12+p*A*A/12
    T_A=q("41/4");H_A=-T/3
    K_A=2*H*H_A-T*T_A/6+p*A/6
    return H,T,K,H_A,T_A,K_A
def gamma(mu,em):
    d=2-mu*em;assert lo(d)>0
    return 21-1/d
def init(al,ah,mu,tan):
    A=E.ball(al,ah);H,T,K,Ha,Ta,Ka=base(A)
    c=(al+ah)/2;K=cap(K,base(arb(c))[2]+Ka*(A-c))
    assert q(".00028")<lo(K)<hi(K)<q(".00032") and lo(Ka)>0 and hi(T)<-q(".018")
    r=1/T;rA=-Ta*r*r;tp=p*A-4*H*T;tpa=p-4*(Ha*T+H*Ta)
    B=gamma(mu,arb(1))-2*tp*r;Ba=-2*(tpa*r+tp*rA);assert lo(B)>0
    lam=mu+4*K;lama=4*Ka
    V=-2*lam*r*r/B
    Va=-2*(lama*r*r/B+2*lam*r*rA/B-lam*r*r*Ba/(B*B))
    z=[arb(0),H,A,T,arb(1),V]
    if tan:z.extend([arb(0),Ha,arb(1),arb(Ta),arb(0),Va])
    return z,K,Ka
def rhs(K,Ka,mu,tan):
    lam=mu+4*K;lama=4*Ka
    def f(z):
        W,H,P,T,y,j=z[:6];em=(-2*W).exp();ap=(2*W).exp();r=inverse(T)
        m=ap*T*T;N=em*em*r*r;B=2*em/3-lam*N
        out=[H,-T*T/3-K*em,T,p*P-4*H*T,m*j,B*y]
        if tan:
            w,h,u,t,Y,J=z[6:]
            Ma=2*ap*T*(W*0+t+T*w)
            Na=-4*N*w-2*em*em*r*r*r*t
            Ba=-4*em*w/3-lama*N-lam*Na
            out.extend([h,-2*T*t/3-Ka*em+2*K*em*w,t,p*u-4*(h*T+H*t),m*J+Ma*j,B*Y+Ba*y])
        return out
    return f
def event(z,K):
    W,H,P,T=z[:4];d=P-cb;X=T+d/4;tp=p*P-4*H*T
    return 6*H-lb-d*d/4-X*X/10,-2*T*X-6*K*(-2*W).exp()-X*(tp+T/4)/5
class Cell:
    def __init__(self,s,h,y,z):
        self.s,self.h,self.z=s,h,z
        n=E.TAYLOR_ORDER
        self.c=[a+[b[n]] for a,b in zip(E.taylor_coefficients(y,n-1),E.taylor_coefficients(z,n))]
    def eval(self,l,h):
        assert self.s<=l<=h<=self.s+self.h
        t=E.ball(l-self.s,h-self.s);res=[]
        for a,b in zip(self.c,self.z):
            x=arb(0)
            for j in reversed(a):x=j+t*x
            res.append(cap(x,b))
        return res
def process(kind,idx,al,ah):
    started=time.monotonic()
    out={"kind":kind,"index":idx,"A_exact":[str(al),str(ah)],"cpu":list(os.sched_getaffinity(0))}
    try:
        case=C[kind];mu=q(case["mu_test"]);tan=kind=="fold"
        z,K,Ka=init(al,ah,mu,tan);A=z[2]
        E.rhs=rhs(K,Ka,mu,tan)
        h=q(C["step"]);n=int(str(q(C["end"])/h));assert n*h==q(C["end"])
        cells=[];before=q(0);after=None;floor=q(2);maxpic=0
        assert lo(event(z,K)[0])>0
        for i in range(n):
            a=i*h;box,pic=E.picard_box(z,h);maxpic=max(pic,maxpic)
            W,H,P,T,Y,V=box[:6];em=(-2*W).exp()
            assert all(x.is_finite() for x in box)
            assert lo(W.exp())>q(".38") and lo(H)>-q("1.2") and hi(H)<-q(".9")
            assert lo(T)>-q(".55") and hi(T)<-q(".017")
            assert lo(P)>-q(".03") and hi(P)<q(".101")
            assert (H*H-1-T*T/12+p*P*P/12-K*em).contains(0)
            if tan:
                w,s,u,t,y,v=box[6:]
                assert (2*H*s-T*t/6+p*P*u/6-(Ka-2*K*w)*em).contains(0)
            assert hi(event(box,K)[1])<0
            assert lo(2-mu*em)>0
            floor=min(floor,lo(Y))
            c=Cell(a,h,z,box);cells.append(c);z=c.eval(a+h,a+h)
            g=event(z,K)[0]
            if lo(g)>0:before=a+h
            if hi(g)<0:after=a+h;break
        assert after is not None and q(".7")<before<after<q("1")
        def evaluate(a,b):
            assert 0<=a<=b<=after
            out=None;cursor=a
            for c in cells:
                l,u=max(a,c.s),min(b,c.s+c.h)
                if l>u:continue
                assert l<=cursor;cursor=max(cursor,u)
                z=c.eval(l,u);out=z if out is None else [E.hull(x,y) for x,y in zip(out,z)]
            assert cursor==b and out is not None
            return out
        dl,dh=before,after
        for _ in range(C["max_event_refinements"]):
            m=(dl+dh)/2;z=evaluate(dl,dh);s=event(z,K)[1];assert hi(s)<0
            g=event(evaluate(m,m),K)[0];N=arb(m)-g/s
            l,u=max(dl,lo(N)),min(dh,hi(N));assert l<=u
            if u-l>=(dh-dl)*q(".999"):
                dl,dh=l,u;break
            dl,dh=l,u
        assert dh-dl<=q(C["max_event_width"])
        z=evaluate(dl,dh);W,H,P,T,Y,V=z[:6];ep=(-2*W).exp();r=1/T
        gg,gs=event(z,K);assert gg.contains(0) and hi(gs)<0
        d=P-cb;X=T+d/4;Xnew=P+X/10
        assert hi(X)<0 and lo(1-(P-cb)/1000)>0 and lo(1-(A-ca)/1000)>0
        gp=gamma(mu,ep);Tp=p*P-4*H*T;B=gp+2*Tp*r;assert lo(B)>0
        assert floor>q(C["positive_y_floor"])
        N=ep*ep*r*r;lam=mu+4*K;R=V-2*lam*N*Y/B
        if case["R_sign"]>0:assert lo(R)>0,("stable sign",pack(R))
        if case["R_sign"]<0:assert hi(R)<0,("unstable sign",pack(R))
        if tan:
            w,h,u,t,y,v=z[6:]
            g_A=6*h-d*u/2-X*(t+u/4)/5;D_A=-g_A/gs
            wt,ht,pt,tt,yt,vt=[a+b*D_A for a,b in zip(z[6:],E.rhs(z)[:6])]
            Na=N*(-4*wt-2*tt*r)
            Tpt=p*pt-4*(ht*T+H*tt)
            Bt=2*(Tpt*r-Tp*r*r*tt) # mu=0 so Gamma is constant
            assert mu==0
            Rp=vt-2*(4*Ka*N*Y/B+lam*Na*Y/B+lam*N*yt/B-lam*N*Y*Bt/(B*B))
            assert hi(Rp)<0,("fold derivative",pack(Rp))
            hD=Tp+q("41/4")*T;Gv=-X/3;assert hi(hD)<0 and lo(Gv)>0
            vA=pt+(tt+pt/4)/10
            Hl,Tl,_,_,_,_=base(A)
            Bal=q("41/2")-2*(p*A-4*Hl*Tl)/Tl
            psia=Tl*Bal/(6*lam)
            F=10-hD*(2*W).exp()*Gv/K
            c=(3* T*B*psia/2)/F
            assert lo(c)>0 and (vA+c*R).contains(0)
            out.update(R_A=pack(Rp),v_A=pack(vA),positive_fold_factor=pack(c),fold_identity_overlap=True)
        out.update(passed=True,k=pack(K),k_A=pack(Ka),D=pack(E.ball(dl,dh)),upper_v=pack(Xnew),
                   R=pack(R),y_floor=str(floor),max_picard=maxpic,cells=len(cells),T_b=pack(T),a_b=pack(W.exp()),
                   test_mu=str(mu),event_cover_exact=[str(before),str(after)],g_s=pack(gs),B_b=pack(B))
    except BaseException as e:
        out.update(passed=False,error=repr(e),traceback=traceback.format_exc())
    out["seconds"]=time.monotonic()-started
    atomic(f"CASE_{kind}_{idx}.json",out)
    return out
def pin():
    cpus=list(map(int,os.environ["STUDY_CPUS"].split(",")));i=MP.current_process()._identity[-1]-1
    os.sched_setaffinity(0,{cpus[i%len(cpus)]});ctx.prec=768;ctx.threads=1
def job(args):return process(*args)
def bounds(d):return tuple(map(q,d["exact"]))
if __name__=="__main__":
    ar=E.Series([arb(2),arb(1),arb(0)]);r=inverse(ar)
    assert all(v.contains(arb(q(x))) for v,x in zip(r.coefficients,("1/2","-1/4","1/8")))
    pilot=sys.argv[2]=="pilot";assert pilot or sys.argv[2]=="full"
    n=C["cells_each"];ids=[0,n//2,n-1] if pilot else list(range(n));work=[]
    for key in ("stable","unstable","fold"):
        a,b=map(q,C[key]["A"])
        work.extend([(key,i,a+(b-a)*i/n,a+(b-a)*(i+1)/n) for i in ids])
        work.extend([(key,"left",a,a),(key,"right",b,b)])
    rows=[];cores=len(os.environ["STUDY_CPUS"].split(","));assert 1<=cores<=6
    with CF.ProcessPoolExecutor(max_workers=cores,mp_context=MP.get_context("fork"),initializer=pin) as ex:
        for r in ex.map(job,work):
            rows.append(r);print(json.dumps({k:r.get(k) for k in ["kind","index","passed","error","seconds"]}),flush=True)
            atomic("progress.json",{"completed":len(rows),"total":len(work),"passed":sum(r["passed"] for r in rows),"last":[r["kind"],r["index"]]})
    ok=all(r["passed"] for r in rows);target=tuple(map(q,C["common_v"]))
    edges={key:{r["index"]:r for r in rows if r["kind"]==key and r["index"] in ("left","right")} for key in ("stable","unstable","fold")}
    if ok:
        assert bounds(edges["stable"]["left"]["upper_v"])[0]>target[1]>target[0]>bounds(edges["stable"]["right"]["upper_v"])[1]
        assert bounds(edges["unstable"]["right"]["upper_v"])[0]>target[1]>target[0]>bounds(edges["unstable"]["left"]["upper_v"])[1]
        assert bounds(edges["fold"]["left"]["R"])[0]>0>bounds(edges["fold"]["right"]["R"])[1]
    result={"all_cases_pass":ok,"full_cover":not pilot,"certified":ok and not pilot,
            "rows":rows,"common_v":C["common_v"],"stable_gap_test":C["stable"]["mu_test"],"cells_each":n,
            "scope":"Whole parameter slabs only; a pilot never proves continuum, same-action or unique fold."}
    atomic("VEV_COEXISTENCE_RESULT.json",result);print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
    assert ok
