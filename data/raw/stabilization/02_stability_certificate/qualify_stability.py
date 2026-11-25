"""Native exact controls for the obstruction and two explicit stability proofs."""
import hashlib,json,os,platform,sys
from pathlib import Path
from fractions import Fraction as Q
from flint import arb,ctx,fmpq
import sympy as S
assert sys.platform=="linux";ctx.prec=768;ctx.threads=1
C=json.loads(Path(sys.argv[1]).read_text())
p,v0,v1,la,lb=[Q(C[k]) for k in ["original_p","v0","v1","lambda_a","lambda_b"]]
q=lambda x:Q(str(x))
checks={}
def ck(k,v):checks[k]=bool(v);assert checks[k],k
def ball(x):return arb(fmpq(str(x)))
ck("exact_source_rounded_bounds",q(".409")<p<q(".411") and q(".1")<=v0<q(".1000001") and q(".0904")<v1<q(".0905") and q("5.994")<la<6 and q("-6.054")<lb<q("-6.0528"))
hmax=q("1.009");a0=(v0+v1)/2
ck("positive_p_monotone_start_rejection",p*v0+4*q(".999")*10*(a0-v0)<0)
ck("H_upper_bound",abs(lb)/6<hmax)
ck("lower_curvature_bound",p*v0*v0/12<q(".00035") and 5*la/9-Q(25,3)+Q(25,9)*(v0-a0)**2<0)
db=v0-v1
ck("upper_curvature_bound",db<q(".0097") and (q("6.0528")-10*q(".0097")**2)**2/36-1-100*q(".0097")**2/12>q(".016"))
ck("duration_lower_exp",(-ball(2*hmax*q("1.89"))).exp().lower()>ball(q(".00035")/q(".016")))
ck("integrated_monotone_contradiction",p*v1*q("1.89")>(5+2*hmax)*(v0-v1))
ck("stronger_p_box_contradiction",q(".4")*v1*q("1.89")>(5+2*hmax)*(v0-v1) and q(".42")*v0*v0/12<q(".000351") and (-ball(2*hmax*q("1.89"))).exp().lower()>ball(q(".000351")/q(".016")))
Alo,Ahi=map(Q,C["two_tension_A_range"]);kl,kh=map(Q,C["two_tension_k_range"])
tl,th=10*(Alo-v0),10*(Ahi-v0)
ck("two_tension_launch_H",1+th**2/12-p*Ahi*Ahi/12+kl>q(".999")**2 and 1+tl**2/12-p*Alo*Alo/12+kh<q("1.001")**2)
tp_l=p*q(".093")-4*q("1.01")*q(".05");tp_h=p*q(".097")-4*q(".99")*q(".035")
ck("two_tension_box_Tprime",q("-.165")<tp_l<tp_h<q("-.098"))
ck("two_tension_box_invariance",tl-q(".165")*q(".05")>q("-.05") and th<q("-.035") and Alo-q(".05")**2>q(".093") and Ahi<q(".097") and -q("1.01")*q(".05")>q("-.06") and q(".05")*(q(".05")**2/3+2*kh)<q(".009"))
ck("exponent_factor_bound",ball(q(".12")).exp().upper()<2)
ck("two_tension_event_transverse",tp_l-10*q(".05")>q("-.665") and tp_h-10*q(".035")<q("-.448"))
ck("two_tension_event_endpoint_signs",20*Alo-10*(v0+v1)-q(".665")*q(".02")>0 and 20*Ahi-10*(v0+v1)-q(".448")*q(".04")<0)
ck("two_tension_warp_and_t_slope",1-q("1.01")*q(".04")>q(".9") and 4*q("1.01")<5)
ck("two_tension_norm_coefficients",4*q(".9")**4*q(".035")**2>q(".003") and 5*q(".9")**4*q(".035")**2>q(".0039"))
ck("two_tension_trace_bound",ball(Q(1,1)/q(".12")).sqrt().upper()+ball(q(".04")).sqrt().upper()*ball(q(".05")/3)<3)
ck("two_tension_positive_complete_gap",162/q(".0039")+6/q(".003")<50000 and Q(1,51000)-4*kh>q(".000015"))
ck("two_tension_E_exit",1-q(".9")**2*q(".035")**2/(6*kh)<0)
ck("two_tension_trial_not_a_certificate",(Q(2,3)*q(".9")**2*q(".035")**2-4*kh)>0)
c,k,x=S.symbols("c k x",positive=True)
a=1/S.cosh(x);H=-c*S.sinh(x);T=S.cosh(x);V=6+6*c*c-S.Rational(3,2)*S.cosh(x)**2
d=lambda z:S.diff(z,x)*c*T
ck("cosine_bulk_H",S.simplify(d(H)+T*T/3+k/a**2).subs(k,c*c-S.Rational(1,3)).simplify()==0)
ck("cosine_bulk_T",S.simplify(d(T)-c*S.diff(V,x)+4*H*T)==0)
ck("cosine_constraint",S.simplify(H*H-1-T*T/12+V/6-k/a**2).subs(k,c*c-S.Rational(1,3)).simplify()==0)
ck("cosine_warp_k_range",1-(Q(1,3)+q(".05"))/2>q(".8"))
ck("cosine_trace_bound",ball(Q(1,6)).sqrt().upper()+ball(Q(1,3))<ball(Q(3,4)))
B=2*q("2.25")**2/(10*q(".8")**2)+6/(4*q(".8")**2)
ck("cosine_gap_mixed_allocated",B+Q(1,40)<4 and Q(1,4)-4*q(".05")==Q(1,20))
ck("cosine_trial_and_cone_exit",Q(2,3)-4*q(".05")>0 and 1-1/(6*q(".05"))<0)
# Pointwise check of the physical shifted-square expansion, including T''.
T,Tp,H,a,V2,k,j,jp,psi=S.symbols("T Tp H a V2 k j jp psi",real=True)
dlt=psi-jp/2;L=-Tp*j+2*T*dlt
Qv=V2+4*T*T/3+4*k/(a*a);Tpp=Qv*T-4*H*Tp
density=a**4*(L**2+Qv*T*T*j*j+4*Tp*T*j*psi)
derivative=a**4*((4*H*T*Tp+Tp*Tp+T*Tpp)*j*j+2*T*Tp*j*jp)
ck("shifted_square_and_boundary_sign",S.expand(density-derivative-4*a**4*T*T*dlt**2)==0)
y,dy=S.symbols("y dy")
ck("arbitrary_profile_norm_identity",S.expand(3*(a*a*T*T*j/3*j+y*jp)+6*y*(y/a**2-jp/2)-(a*a*T*T*j*j+6*y*y/a**2))==0)
f,u=S.symbols("f u",real=True);P=(u-f/2)**2/2
ck("mixed_wall_hessian",S.hessian(P,(f,u))==S.Matrix([[S.Rational(1,4),-S.Rational(1,2)],[-S.Rational(1,2),1]]))
ck("adverse_pure_psi_trial_sign", (Q(2,3)*Q(1,10)**2-4*Q(1,100))<0)
out={"pass":True,"checks":checks,"count":len(checks),"runtime":{"python":sys.version,"platform":platform.platform(),"cpus":sorted(os.sched_getaffinity(0)),"precision":ctx.prec,"flint":__import__('flint').__version__},"config_sha256":hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()}
Path("QUALIFICATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"pass":True,"count":len(checks)}))
