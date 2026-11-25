"""Refined physical fold and finite two-mass response bound, native only."""
import hashlib,json,os,sys
from pathlib import Path
from flint import arb
import vev_coexistence as V
E=V.E;q=V.q;lo=V.lo;hi=V.hi;pack=V.pack
assert sys.platform=="linux"
C=json.loads(Path(sys.argv[2]).read_text());raw=Path(C["source_native"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==C["source_sha256"]
S=json.loads(raw);assert S["certified"] and S["all_cases_pass"] and S["full_cover"]
rows=[r for r in S["rows"] if r["kind"]=="fold" and isinstance(r["index"],int)]
assert len(rows)==V.C["cells_each"]
def bounds(d):return tuple(map(q,d["exact"]))
F_l=min(-bounds(r["R_A"])[1] for r in rows);F_h=max(-bounds(r["R_A"])[0] for r in rows)
c_l=min(bounds(r["positive_fold_factor"])[0] for r in rows);c_h=max(bounds(r["positive_fold_factor"])[1] for r in rows)
assert 0<F_l<=F_h and 0<c_l<=c_h
der=E.ball(-F_h,-F_l)
a0,b0=map(q,V.C["fold"]["A"]);a,b=a0,b0
history=[]
for j in range(C["max_refinements"]):
    m=(a+b)/2;s=V.process("fold","refine"+str(j),m,m);assert s["passed"],s
    R=E.ball(*bounds(s["R"]));N=arb(m)-R/der
    a,b=max(a,lo(N)),min(b,hi(N));assert a<=b
    history.append({"iteration":j,"point":str(m),"R":s["R"],"A_new":[str(a),str(b)]})
    V.atomic("FOLD_REFINEMENT.json",history)
    if b-a<=q(C["A_width_target"]):break
assert b-a<=q(C["A_width_target"])
cent=(a+b)/2;sc=V.process("fold","exactcenter",cent,cent);assert sc["passed"]
v0,v1=bounds(sc["upper_v"]);correction=c_h*F_h*(b-a)**2/8
v_l,v_h=v0-correction,v1
eta,r=q(C["A_radius"]),q(C["mass_radius"])
al,ah=a-eta,b+eta;assert a0<al<ah<b0
mu=E.ball(-r,r)
y,K,Ka=V.init(al,ah,mu,False);launch=y.copy()
V.E.rhs=V.rhs(K,Ka,mu,False)
h=q(V.C["step"]);n=int(str(q(V.C["end"])/h));cells=[];before=q(0);after=None;minY=q(2);maxY=q(0)
for i in range(n):
    t=i*h;z,pic=E.picard_box(y,h)
    W,H,P,T,Y,F=z;em=(-2*W).exp()
    assert all(x.is_finite() for x in z)
    assert (H*H-1-T*T/12+V.p*P*P/12-K*em).contains(0)
    assert lo(T)>-q(".55") and hi(T)<-q(".017") and lo(W.exp())>q(".38")
    assert hi(V.event(z,K)[1])<0 and lo(2-mu*em)>0
    minY,maxY=min(minY,lo(Y)),max(maxY,hi(Y))
    c=V.Cell(t,h,y,z);cells.append(c);y=c.eval(t+h,t+h)
    g=V.event(y,K)[0]
    if lo(g)>0:before=t+h
    if hi(g)<0:after=t+h;break
assert after is not None and q(".7")<before<after<q(1) and minY>q("1/2")
def eval_time(l,u):
    cursor=l;out=None
    for c in cells:
        a,b=max(l,c.s),min(u,c.s+h)
        if a>b:continue
        assert a<=cursor;cursor=max(cursor,b)
        z=c.eval(a,b)
        out=z if out is None else [E.hull(x,y) for x,y in zip(out,z)]
    assert cursor==u and out is not None
    return out
dl,dh=before,after
for _ in range(V.C["max_event_refinements"]):
    m=(dl+dh)/2;v=V.event(eval_time(m,m),K)[0];sl=V.event(eval_time(dl,dh),K)[1];assert hi(sl)<0
    N=arb(m)-v/sl;l,u=max(dl,lo(N)),min(dh,hi(N));assert l<=u
    if u-l>=(dh-dl)*q(".999"):dl,dh=l,u;break
    dl,dh=l,u
assert dh-dl<q(".005")
z=eval_time(dl,dh)
yb_l,yb_h=E.bounds(z[4]);assert 0<yb_l<=yb_h
I_l=I_h=q(0)
for c in cells:
    l=max(q(0),min(h,dl-c.s));u=max(q(0),min(h,dh-c.s))
    if not u:continue
    W,H,P,T,Y,F=c.z;ww=(-4*W).exp()/(T*T)
    yl,yh=E.bounds(Y);assert yl>0 and lo(ww)>0
    I_l+=l*lo(ww)*yl*yl;I_h+=u*hi(ww)*yh*yh
def Fprime(y,sgn):
    W,H,P,T,Y,F=y;em=(-2*W).exp();d=2-mu*em
    G=21-1/d;B=G+sgn*2*(V.p*P-4*H*T)/T
    assert lo(d)>0 and lo(B)>0
    gp=-em/(d*d);omega=2*em*em/(T*T*B);lam=mu+4*K;assert lo(lam)>0
    x=omega-lam*omega*gp/B
    assert lo(x)>0
    return E.bounds(x)
a_l,a_h=Fprime(launch,-1);b_l,b_h=Fprime(z,1)
d_l=(I_l+a_l)/(yb_h*yb_h)+b_l;d_h=(I_h+a_h)/(yb_l*yb_l)+b_h
assert 0<d_l<=d_h
tau=min(eta,r*d_l*yb_l/F_h);v_cap=c_l*F_l*tau*tau/2
assert v_cap>=q(C["v_radius_target"])
l=lo(arb(F_l/(d_h*yb_h))*arb(2/(c_h*F_h)).sqrt())
u=hi(arb(F_h/(d_l*yb_l))*arb(2/(c_l*F_l)).sqrt());assert 0<l<u
delta=v_l+v_cap-v_h
absolute=[v_h+delta/4,v_h+3*delta/4] if delta>0 else None
out={"pass":True,"fold_A":{"exact":[str(a),str(b)],"display":[float(a),float(b)]},
     "fold_v":{"exact":[str(v_l),str(v_h)],"display":[float(v_l),float(v_h)]},
     "A_band_exact":[str(al),str(ah)],"D_band":pack(E.ball(dl,dh)),
     "positive_y_all_times":pack(E.ball(minY,maxY)),"yb":pack(E.ball(yb_l,yb_h)),
     "cross_norm_bulk":pack(E.ball(I_l,I_h)),"normalized_residual_derivative_magnitude":pack(E.ball(d_l,d_h)),
     "v_radius_exact":str(v_cap),"v_radius_display":float(v_cap),"sqrt_coeff":pack(E.ball(l,u)),
     "absolute_v_interval_exact":[str(v) for v in absolute] if absolute else None,
     "absolute_v_interval_display":[float(v) for v in absolute] if absolute else None,
     "scope":"Finite same-action square-root spectral law for both local static branches; original phase zeros and original kinetic sector retained.",
     "source_sha256":C["source_sha256"],"cpus":sorted(os.sched_getaffinity(0))}
V.atomic("VEV_FOLD_FINITE.json",out)
print(json.dumps({k:out[k]["display"] for k in ("fold_A","fold_v","sqrt_coeff","normalized_residual_derivative_magnitude")}|{"v_radius":float(v_cap),"absolute_interval":out["absolute_v_interval_display"]}))
