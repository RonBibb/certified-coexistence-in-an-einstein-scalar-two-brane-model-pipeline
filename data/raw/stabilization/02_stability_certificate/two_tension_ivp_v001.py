"""Native validated original-bulk shooting and tension-coordinate inverse."""
from pathlib import Path
import json,os,sys,hashlib
from flint import arb,fmpq,ctx
import ivp_engine as E
assert sys.platform=="linux";ctx.prec=768;ctx.threads=1
C=json.loads(Path(sys.argv[1]).read_text());L=json.loads(Path(C["control_literals"]).read_text())
q=lambda v:fmpq(str(v))
p,v0,v1=map(q,[L[x] for x in ("original_p","v0","v1")])
lo=lambda a:E.bounds(a)[0]
hi=lambda a:E.bounds(a)[1]
def pack(a):
    x,y=E.bounds(a);return {"exact":[str(x),str(y)],"display":[float(x),float(y)]}
def sq(a):
    x,y=E.bounds(a);return E.ball(q(0) if x<=0<=y else min(x*x,y*y),max(x*x,y*y))
def intersect(a,b):
    l,h=max(lo(a),lo(b)),min(hi(a),hi(b));assert l<=h
    return E.ball(l,h)
def rhs(k,variational):
    def f(z):
        W,H,P,T=z[:4]
        a2=(-2*W).exp()
        out=[H,-T*T/3-k*a2,T,p*P-4*H*T]
        if variational:
            for off,kp in [(4,0),(8,1)]:
                w,h,x,t=z[off:off+4]
                out.extend([h,-2*T*t/3-kp*a2+2*k*a2*w,t,p*x-4*(h*T+H*t)])
        return out
    return f
def launch(A,k):
    T=10*(A-v0);H=-(1+sq(T)/12-p*sq(A)/12+k).sqrt()
    Ha=(10*T-p*A)/(12*H);Hk=1/(2*H)
    return [arb(0),H,A,T,arb(0),Ha,arb(1),arb(10),arb(0),Hk,arb(0),arb(0)]
def constraint(z,k,variational):
    W,H,P,T=z[:4];e=(-2*W).exp()
    assert (H*H-1-T*T/12+p*P*P/12-k*e).contains(0)
    if variational:
        for off,kp in [(4,0),(8,1)]:
            w,h,x,t=z[off:off+4]
            assert (2*H*h-T*t/6+p*P*x/6-kp*e+2*k*e*w).contains(0)
class Cell:
    def __init__(self,y,z,s,h):
        self.z=z;self.s=s;self.h=h
        n=E.TAYLOR_ORDER
        self.c=[a+[b[n]] for a,b in zip(E.taylor_coefficients(y,n-1),E.taylor_coefficients(z,n))]
    def eval(self,l,h):
        assert self.s<=l<=h<=self.s+self.h
        dt=E.ball(l-self.s,h-self.s);out=[]
        for row,b in zip(self.c,self.z):
            y=arb(0)
            for a in reversed(row):y=a+dt*y
            out.append(intersect(y,b))
        return out

def shoot(name,A,k):
    E.rhs=rhs(k,True);y=launch(A,k);initial=y.copy()
    h=q(C["step"]);n=int(str(q("1/25")/h));assert n*h==q("1/25")
    cells=[];before=q(0);after=None;max_pic=0
    for i in range(n):
        t=i*h;z,pic=E.picard_box(y,h);max_pic=max(max_pic,pic);constraint(z,k,True)
        assert lo(z[3])>q("-.05") and hi(z[3])<q("-.035")
        assert lo(z[1])>q("-1.01") and hi(z[1])<q("-.99") and lo(z[0].exp())>q(".9")
        dv=E.rhs(z)[3]+10*z[3]
        assert q("-.665")<lo(dv)<hi(dv)<q("-.448")
        c=Cell(y,z,t,h);cells.append(c);y=c.eval(t+h,t+h)
        value=y[3]+10*(y[2]-v1)
        if lo(value)>0:before=t+h
        if hi(value)<0:
            after=t+h;break
    assert after is not None and q(".02")<=before<after<=q(".04")
    def evaluate(dl,dh):
        assert 0<=dl<=dh<=after
        found=None;cursor=dl
        for c in cells:
            l,h=max(dl,c.s),min(dh,c.s+c.h)
            if l>h:continue
            assert l<=cursor;cursor=max(cursor,h)
            z=c.eval(l,h)
            found=z if found is None else [E.hull(x,y) for x,y in zip(found,z)]
        assert cursor==dh and found is not None
        return found
    dl,dh=before,after
    for _ in range(12):
        s=(dl+dh)/2;z=evaluate(dl,dh);m=evaluate(s,s)
        Qs=E.rhs(z)[3]+10*z[3];assert hi(Qs)<0
        N=arb(s)-(m[3]+10*(m[2]-v1))/Qs
        l,h=max(dl,lo(N)),min(dh,hi(N));assert l<=h
        if h-l>=(dh-dl)*q("999/1000"):
            dl,dh=l,h;break
        dl,dh=l,h
    y=evaluate(dl,dh);W,H,B,T=y[:4];dv=E.rhs(y)[3]+10*T
    assert (T+10*(B-v1)).contains(0)
    LA=-6*initial[1]-10*sq(A-v0);LB=6*H-10*sq(B-v1)
    J=[[],[]]
    for off,kx in [(4,0),(8,1)]:
        s=-(y[off+3]+10*y[off+2])/dv
        jlo=-6*initial[off+1]-(2*initial[3] if kx==0 else 0)
        # At the actual scalar event, 6H'+2T²=-6k/a².
        jhi=6*y[off+1]+2*T*y[off+2]-6*k*(-2*W).exp()*s
        J[0].append(jlo);J[1].append(jhi)
    out={"name":name,"A":pack(A),"k":pack(k),"D":pack(E.ball(dl,dh)),
         "event_sign_bracket_exact":[str(before),str(after)],"flow_cells":len(cells),
         "max_picard":max_pic,"all_first_integrals_and_tangents":True,
         "lambda_a":pack(LA),"lambda_b":pack(LB),"B":pack(B),"T_b":pack(T),
         "a_b":pack(W.exp()),"Jacobian": [[pack(x) for x in row] for row in J]}
    return out,J

def atomic(name,value):
    tmp=Path(name+".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(value,indent=2)+"\n");f.flush();os.fsync(f.fileno())
    tmp.replace(name)

if __name__=="__main__":
    Ac,kc=q(C["A_center"]),q(C["k_center"]);ra,rk=q(C["A_inverse_radius"]),q(C["k_inverse_radius"])
    a_range,k_range=[list(map(q,L[k])) for k in ("two_tension_A_range","two_tension_k_range")]
    assert a_range[0]<Ac-ra<Ac+ra<a_range[1] and k_range[0]<kc-rk<kc+rk<k_range[1]
    rows=[]
    for tag,A,k in [("target",arb(Ac),arb(q(C["target_k"]))),("inverse_center",arb(Ac),arb(kc)),
                    ("inverse_box",E.ball(Ac-ra,Ac+ra),E.ball(kc-rk,kc+rk))]:
        r,J=shoot(tag,A,k);rows.append(r);atomic("CASE_"+tag+".json",r)
        atomic("progress.json",{"completed":len(rows),"total":3,"last_durable_case":tag})
    J=[[x*f for x,f in zip(row,[ra,rk])] for row in J]
    mid=[[(lo(x)+hi(x))/2 for x in row] for row in J]
    a,b=mid[0];c,d=mid[1];det=a*d-b*c;assert det!=0
    R=[[d/det,-b/det],[-c/det,a/det]]
    defect=[[arb(int(i==j))-sum((R[i][l]*J[l][j] for l in range(2)),arb(0)) for j in range(2)] for i in range(2)]
    dn=max(sum(max(abs(lo(x)),abs(hi(x))) for x in row) for row in defect)
    # Preserve exact center and determinant bounds even if the inverse radius fails.
    Rnorm=max(sum(abs(x) for x in row) for row in R)
    eps=1/(4*Rnorm)
    passed=dn<q(C["expected_preconditioner_defect_upper"])
    out={"pass":passed,"rows":rows,"inverse_box_scaled_preconditioner":[list(map(str,row)) for row in R],
         "defect_upper_exact":str(dn),"defect_display":float(dn),"tension_inf_radius_exact":str(eps),
         "tension_inf_radius_display":float(eps),"scope":"An independent additive-tension neighborhood around exact interior center if inverse bound passes; background depends on target action; original p, VEVs and stiffness fixed."}
    atomic("TENSION_IVP.json",out)
    print(json.dumps({"pass":passed,"defect":float(dn),"tension_radius":float(eps),"cases":[{k:r[k]["display"] for k in ("lambda_a","lambda_b","D","B","a_b")} for r in rows]}))
    assert passed
