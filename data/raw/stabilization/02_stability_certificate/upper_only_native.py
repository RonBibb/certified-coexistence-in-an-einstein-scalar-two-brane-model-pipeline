"""Independent original-bulk qualification and undivided physical low-mass shooting."""
import hashlib,json,os,sys
from pathlib import Path
from fractions import Fraction
from flint import arb,fmpq,ctx
import ivp_engine as E
assert sys.platform=="linux";ctx.prec=768;ctx.threads=1
C=json.loads(Path(sys.argv[1]).read_text());C0=json.loads(Path(C["control_literals"]).read_text())
q=lambda x:fmpq(str(Fraction(str(x))))
p,v0,v1,la,lb=[q(C0[j]) for j in ("original_p","v0","v1","lambda_a","lambda_b")]
k,D=q(C["k"]),q(C["D"]);h=D/C["steps"];lo=lambda a:E.bounds(a)[0];hi=lambda a:E.bounds(a)[1]
def pack(x):
    l,h=E.bounds(x);return {"exact":[str(l),str(h)],"display":[float(l),float(h)]}
def intersect(x,y):
    l,h=max(lo(x),lo(y)),min(hi(x),hi(y));assert l<=h
    return E.ball(l,h)
def qiv(x):return E.ball(*map(q,x))
raw=Path(C["family_source_native"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==C["family_source_sha256"]
OLD=json.loads(raw);assert OLD["certified"] and OLD["inherited_exact_anchor_in_common_box"]
left,right=[OLD["rows"][i] for i in (319,320)]
assert q(left["p_exact"][1])==p==q(right["p_exact"][0]) and left["pass"] and right["pass"]
astar=intersect(qiv(C["A_star_manuscript_exact"]),intersect(qiv(left["newton_exact_raw"]),qiv(right["newton_exact_raw"])))
bstar=intersect(qiv(left["root_image_geometry"]["quantities"]["Phi"]["exact_bounds"]),
                qiv(right["root_image_geometry"]["quantities"]["Phi"]["exact_bounds"]))

def atomic(name,d):
    tmp=Path(name+".tmp")
    with tmp.open("w") as f:f.write(json.dumps(d,indent=2)+"\n");f.flush();os.fsync(f.fileno())
    tmp.replace(name)

def lower(A,mix):
    dx=A-v0;dc=A-astar
    H=-(la+10*dx*dx+mix*dc*dc/4)/6
    T=10*dx+mix*dc/4
    K=H*H-1-T*T/12+p*A*A/12
    KA=2*H*(-T/3)-T*(10+q(1)*mix/4)/6+p*A/6
    return H,T,K,KA

def initial(mix):
    l,u=map(q,C["launch_bracket"]);A=E.ball(l,u)
    ql=lower(arb(l),mix)[2];qu=lower(arb(u),mix)[2]
    assert hi(ql)<k<lo(qu) and lo(lower(A,mix)[3])>0
    proof={"initial_K_lo":pack(ql),"initial_K_hi":pack(qu),"initial_K_derivative":pack(lower(A,mix)[3]),"newton":[]}
    for j in range(16):
        c=(lo(A)+hi(A))/2;den=lower(A,mix)[3];assert lo(den)>0
        N=arb(c)-(lower(arb(c),mix)[2]-k)/den
        B=intersect(A,N)
        proof["newton"].append({"iteration":j,"A":pack(B)})
        if hi(B)-lo(B)>=(hi(A)-lo(A))*q("99/100"):
            A=B;break
        A=B
    H,T,K,KA=lower(A,mix)
    assert K.contains(arb(k)) and lo(T)>-q("83/1000") and hi(T)<-q("7/100")
    assert -q("101/100")<lo(H)<hi(H)<-q("99/100")
    return [arb(0),H,A,T],proof

def wall_eff(a_inv2,mu,mix):
    return arb(20) if not mix else 21-1/(2-mu*a_inv2)
def rhs(mu,with_mass):
    def ff(z):
        W,H,A,T=z[:4];em2=(-2*W).exp()
        Tp=p*A-4*H*T
        out=[H,-T*T/3-k*em2,T,Tp]
        if with_mass:
            psi,f,L=z[4:]
            out.extend([-2*H*psi-T*f/3,L-2*T*psi,-4*H*L+(p+4*T*T/3-mu*em2)*f-2*Tp*psi])
        return out
    return ff
def physical_flow(y0,mix,mu=None):
    y=y0.copy();with_mass=mu is not None;M=mu if with_mass else arb(0)
    if with_mass:
        Ga=wall_eff(arb(1),M,mix);H,A,T=y[1:]
        assert lo(2-M)>0
        Tp=p*A-4*H*T;den=T*Ga/2-Tp;assert hi(den)<0
        f=3*(M+4*k)/den;y.extend([arb(1),f,Ga*f/2])
    E.rhs=rhs(M,with_mass);mins={};max_pic=0
    for j in range(C["steps"]):
        z,pic=E.picard_box(y,h);max_pic=max(max_pic,pic)
        W,H,A,T=z[:4];em2=(-2*W).exp();tp=p*A-4*H*T
        assert (H*H-1-T*T/12+p*A*A/12-k*em2).contains(0)
        assert lo(W.exp())>q("3/4") and lo(H)>-q("11/10") and hi(H)<-q("9/10")
        assert lo(A)>0 and hi(A)<q("1/10") and lo(T)>-q("1/4") and hi(T)<-q("1/20")
        assert lo(tp/T)>0 and hi(tp/T)<q("22/5")
        qty={"warp":W.exp(),"T_abs":-T,"ratio":tp/T}
        if with_mass:
            psi,f,L=z[4:]
            assert (T*L-tp*f-3*(M+4*k)*em2*psi).contains(0)
            assert lo(2-M*em2)>0
            qty["metric_y"]=(2*W).exp()*psi
        for key,x in qty.items():
            if key not in mins:mins[key]=[lo(x),hi(x)]
            else:mins[key]=[min(mins[key][0],lo(x)),max(mins[key][1],hi(x))]
        y=E.advance(y,z,h)
    return y,{k:{"exact":list(map(str,lr)),"display":list(map(float,lr))} for k,lr in mins.items()},max_pic

def mode(y0,mix,mu):
    y,bound,pic=physical_flow(y0,mix,mu)
    W,H,A,T,psi,f,L=y
    G=wall_eff((-2*W).exp(),mu,mix)
    upper=L+G*f/2
    return upper,bound,pic

def background(mix):
    y0,proof=initial(mix);y,bounds,pic=physical_flow(y0,mix)
    W,H,B,T=y
    v=B+T/10+mix*(B-bstar)/40
    lam=6*H-10*(B-v)*(B-v)-mix*(B-bstar)*(B-bstar)/4
    actual=lam+10*(B-v)*(B-v)+mix*(B-bstar)*(B-bstar)/4
    slope=20*(B-v)+mix*(B-bstar)/2
    assert (actual-6*H).contains(0) and (slope+2*T).contains(0)
    z=q(3)/4;Tmin=q(1)/20
    ct=2*(q(25)/8)**2/(5*z**4*Tmin**2)+6/(4*z**4*Tmin**2)
    gap=1/q(7000)-4*k
    assert ct<7000 and gap>q(C["target_gap_round"])
    result={"pass":True,"mix":mix,"root_proof":proof,"launch_A":pack(y0[2]),"k":str(k),"D":str(D),
            "upper_Phi":pack(B),"upper_T":pack(T),"upper_H":pack(H),"v1_new":pack(v),"lambda_b_new":pack(lam),
            "lower_chi":pack(mix*(y0[2]-astar)/2),"upper_chi":pack(mix*(B-bstar)/2),
            "upper_metric_scalar_residual_containment":True,"whole_cell_bounds":bounds,"picard_max":pic,
            "gap_exact":str(gap),"original_centers_A_B":[pack(astar),pack(bstar)]}
    return y0,result

if __name__=="__main__":
    phase=sys.argv[2];assert phase in ("background","mass")
    results=[]
    for mix in (0,1):
        y0,r=background(mix);atomic(f"BACKGROUND_{mix}.json",r)
        if phase=="mass":
            ml,mh=map(q,C["mass_bracket"]);hist=[]
            for m in (ml,mh):
                x,_,_=mode(y0,mix,arb(m));hist.append({"mu":str(m),"residual":pack(x)})
            f0,f1=[qiv(x["residual"]["exact"]) for x in hist]
            assert (lo(f0)>0>hi(f1) or lo(f1)>0>hi(f0)),"initial strict mass sign bracket"
            positive=lo(f0)>0
            for j in range(C["bisections_max"]):
                if mh-ml<=q(C["mass_width_target"]):break
                m=(ml+mh)/2;x,_,_=mode(y0,mix,arb(m))
                hist.append({"mu":str(m),"residual":pack(x)})
                if lo(x)<=0<=hi(x):break
                if (lo(x)>0)==positive:ml=m
                else:mh=m
            atomic(f"MASS_HISTORY_{mix}.json",hist)
            assert mh-ml<=q(C["mass_width_target"]),"finite width"
            _,yband,_=mode(y0,mix,E.ball(ml,mh))
            assert q(yband["metric_y"]["exact"][0])>0
            r["mass_bracket"]={"exact":[str(ml),str(mh)],"display":[float(ml),float(mh)]}
            r["whole_mass_bracket_metric_y"]=yband["metric_y"]
            r["lowest_root_proof"]="Strict physical upper sign change, positive metric y on full bracket, ground-state identity, monotone mu pencil before wall poles and global coercivity below zero."
        results.append(r);atomic("progress.json",{"completed":len(results),"total":2,"latest":mix})
    out={"pass":True,"phase":phase,"cpus":sorted(os.sched_getaffinity(0)),"rows":results}
    atomic("UPPER_ONLY_RESULT.json",out)
    print(json.dumps({k:v for k,v in out.items() if k!="rows"}|{"summary":[{x:r[x]["display"] for x in ("launch_A","v1_new","lambda_b_new","mass_bracket") if x in r} for r in results]}))
