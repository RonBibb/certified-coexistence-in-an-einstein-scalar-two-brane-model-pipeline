"""Proposal only: native floating sampling of the one-coefficient continuation."""
import sys,hashlib,json,math
from fractions import Fraction as F
from pathlib import Path
assert sys.platform=="linux"
C=json.loads(Path(sys.argv[1]).read_text());B=json.loads(Path("CONFIG_v003_challenge_v001.json").read_text())
raw=Path(B["candidate_native"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==B["candidate_sha256"]
S=json.loads(raw)
L=json.loads(Path("CONFIG_controls_v001.json").read_text())
p,v0,v1,la,lb=[float(F(L[x])) for x in ["original_p","v0","v1","lambda_a","lambda_b"]]
ca,cb=[float(sum(map(F,S["input_source"][n]))/2) for n in ["C_a","C_b"]]
h=float(F(C["h"]));guard=float(F(C["mode_division_guard"]))
def sample(A):
    H=-(la+10*(A-v0)**2+(A-ca)**2/4)/6;T=10*(A-v0)+(A-ca)/4
    k=H*H-1-T*T/12+p*A*A/12
    r=(p*A-4*H*T)/T;B=20.5-2*r
    if k<=0:return {"A":A,"k":k,"error":"nonpositive_launch_k"}
    mu=0.;lam=4*k
    s0=[0.,H,A,T,1.,-2*lam/(T*T*B)] if B>0 else [0.,H,A,T,1.,0.]
    flag=B>0;fy=1.
    def ode(x,mode):
        W,H,A,T=x[:4];e=math.exp(-2*W);tp=p*A-4*H*T
        ans=[H,-T*T/3-k*e,T,tp]
        if mode:
            y,v=x[4:];ans.extend([T*T*v/e,((2/3)*e-lam*e*e/(T*T))*y])
        else:ans.extend([0.,0.])
        return ans
    def g(x):
        W,H,A,T=x[:4];d=A-cb
        return 6*H-lb-d*d/4-(T+d/4)**2/10
    r0=g(s0);turn=False
    for n in range(int(float(F(C["s_end"]))/h)):
        y=s0
        if y[3]>=-guard:flag=False
        try:
            a=ode(y,flag);b=ode([x+h*t/2 for x,t in zip(y,a)],flag)
            c=ode([x+h*t/2 for x,t in zip(y,b)],flag);d=ode([x+h*t for x,t in zip(y,c)],flag)
        except (ValueError,OverflowError,ZeroDivisionError):break
        z=[x+h*(aa+2*bb+2*cc+dd)/6 for x,aa,bb,cc,dd in zip(y,a,b,c,d)]
        r1=g(z)
        if z[3]>=0:turn=True
        if r0>0>r1:
            w=r0/(r0-r1);x=[a+w*(b-a) for a,b in zip(y,z)]
            W,H,P,T,ym,j=x;X=T+(P-cb)/4
            B_b=20.5+2*(p*P-4*H*T)/T if abs(T)>guard else float("nan")
            rr=j-2*lam*math.exp(-4*W)*ym/(T*T*B_b) if flag and not turn and B_b>0 else None
            return {"A":A,"k":k,"D":(n+w)*h,"v":P+X/10,"T_b":T,"a_b":math.exp(W),"turn":turn,
                    "R_zero":rr,"min_y":min(fy,ym) if rr is not None else None,"B_a":B,"B_b":B_b}
        s0,r0=z,r1;fy=min(fy,z[4])
        if max(map(abs,z[:4]))>50:break
    return {"A":A,"k":k,"error":"no_event_in_diagnostic"}
a,b,s=map(F,[C["from"],C["through"],C["step"]]);n=int((b-a)/s)+1
rows=[sample(float(a+i*s)) for i in range(n)]
Path("VEV_DIAGNOSTIC.json").write_text(json.dumps({"certified":False,"rows":rows,"scope":"sampled float proposals only, no count or continuum conclusion"},indent=2)+"\n")
for x in rows:
    if x["A"] in (0.094,0.095,0.096,0.097,0.098,0.099) or (x.get("R_zero") is not None and abs(x["R_zero"])<.05):
        print(json.dumps(x))
