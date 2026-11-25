"""Native exact constrained action audit and extended existing-action screens."""
import json,os,sys,platform
from fractions import Fraction as Q
from pathlib import Path
import sympy as S
assert sys.platform=="linux"
checks={}
def ck(k,v):checks[k]=bool(v);assert checks[k],k
H,T,a,k,V,Vp,Vpp,Tp=S.symbols("H T a k V Vp Vpp Tp",real=True)
u,f,fp,e=S.symbols("u f fp e",real=True)
up=-2*H*u-T*f/3;Hp=-T*T/3-k/a**2
rad=a**4*(S.exp(6*e*u)*(12*(H+e*up)**2-(T+e*fp)**2)+S.exp(2*e*u)*(12-2*(V+e*Vp*f+e**2*Vpp*f**2/2)))
r2=S.diff(rad,e,2).subs(e,0)/2
L=fp+2*T*u
neg=-a**4*(L*L+(Vpp+4*T*T/3)*f*f-4*Tp*f*u-24*k*u*u/a**2)
bd=a**4*(4*H*(48*H*u*u-8*T*u*f)+48*Hp*u*u+96*H*u*up-8*(Tp*u*f+T*up*f+T*u*fp))
ck("ADM_full_restriction_equals_V3_plus_boundary",S.factor((r2-neg-bd).subs(Tp,Vp-4*H*T).subs(V,6*(1+T*T/12+k/a**2-H*H)))==0)
tau,taup,hf=S.symbols("tau taup hf")
wall=S.diff(-a**4*S.exp(4*e*u)*(tau+e*taup*f+e**2*hf/2),e,2).subs(e,0)/2
for sig,label in [(1,"lower"),(-1,"upper")]:
    boundary=-sig*a**4*(48*(-sig*tau/6)*u*u-8*(sig*taup/2)*u*f)
    ck("doubled_GHY_cancels_wall_metric_"+label,S.expand(wall+boundary+a**4*hf/2)==0)
ck("warp_kinetic_cancellation",S.simplify(S.exp(-2*e*u)*S.exp(4*e*u)*S.exp(-2*e*u))==1)
chi,center,v=S.symbols("chi center v")
U=(chi**2+(chi-(f-center))**2)/2
ck("original_stationary_oscillator_graph",S.solve(S.diff(U,chi),chi)==[(f-center)/2])
eff=10*(f-v)**2+U.subs(chi,(f-center)/2)
ck("original_coupled_effective_quadratic",S.expand(eff-S.Rational(41,4)*(f-v-(center-v)/41)**2-S.Rational(10,41)*(center-v)**2)==0)
C=json.loads(Path(sys.argv[1]).read_text());p,v0,v1,la,lb=map(Q,[C[x] for x in ["original_p","v0","v1","lambda_a","lambda_b"]])
d=v0-v1;delta=-la-lb
ck("baseline_drop_and_VEV_gates",Q(".009")<d<Q(".01") and Q(".04")<delta<Q(".064") and v0>Q(".09") and v1>Q(".09"))
ck("decoupled_monotone_pure_psi_strict",10*d*d < 2*(delta-20*d*d)/6)
dg=d+2*Q(".002");dg0=d-2*Q(".002")
dep=delta-2*Q(10,41)*Q(".08")**2
ck("coupled_monotone_pure_psi_strict",dg0>0 and dg<Q(".014") and Q(41,4)*dg*dg<2*(dep-Q(41,2)*dg*dg)/6)
c0=Q("9901/100000")
ck("sextic_off_axis_lower_slope_strict",v1<c0<v0 and 100*(v0-c0)>Q(1,80))
ck("sextic_monotone_pure_psi_strict",10*d*d<2*(delta-20*d*d)/6)
z,beta,zeta=S.symbols("z beta zeta",positive=True);alpha=-beta*z-zeta*z*z
ck("sextic_stationary_tension_drop",S.expand(alpha*z/2+beta*z*z/4+zeta*z*z*z/6+beta*z*z/4+zeta*z*z*z/3)==0)
out={"pass":True,"count":len(checks),"checks":checks,"scope":"restricted ADM action/finite-wall identities and original/sextic global negative-form exclusions","runtime":{"python":sys.version,"platform":platform.platform(),"affinity":sorted(os.sched_getaffinity(0))}}
Path("ACTION_CONTROLS.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"pass":True,"count":len(checks)}))
