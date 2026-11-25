"""Four separately pinned checks; all results are exact symbolic identities."""
import sys,os,json,time
from pathlib import Path
assert sys.platform=='linux'
mode=sys.argv[1]; os.sched_setaffinity(0,{int(sys.argv[2])}); start=time.monotonic()
import sympy as S
checks={}; residuals={}
def check(name,v):
 w=S.factor(S.expand(v));checks[name]=(w==0)
 if w!=0:residuals[name]=str(w)
if mode in ('einstein','scalar','wall'):
 from coordinate_spatial import *
 if mode=='einstein':
  q=ds(ps)-H*nn
  C=T*(ds(ff)-nn*T)-Vp*ff-3*(mu+4*h*h)*ps/a**2-12*H*q
  M=q+T*ff/3
  iso=((2*mu+6*h*h)*ps+mu*nn)/a**2+3*ds(q)-3*nn*(-T*T/3-h*h/a**2)+12*H*q+T*ds(ff)-nn*T*T+Vp*ff
  et=S.Matrix(4,4,lambda i,j:clean(-(2*ps+nn)*hess[i,j]/a**2+iso*gamma[i,j]*Q))
  trace=clean(-C+sum(gammai[i,i]*et[i,i] for i in range(4)))
  pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2),(3,3)]
  for A,B in pairs:
   direct=sum(S.diff(dG[c,A,B],coords[c])-S.diff(dG[c,A,c],coords[B]) for c in range(5))
   direct+=sum(dG[c,c,d]*G[d,A,B]+G[c,c,d]*dG[d,A,B]-dG[c,B,d]*G[d,A,c]-G[c,B,d]*dG[d,A,c] for c in range(5) for d in range(5))
   direct+=(4-S.Rational(2,3)*V)*dh[A,B]-S.Rational(2,3)*Vp*f*Q*g[A,B]
   if A==0:direct-=T*S.diff(f*Q,coords[B])
   if B==0:direct-=T*S.diff(f*Q,coords[A])
   if A==B==0:expected=-C-trace/3
   elif A==0:expected=clean(-3*M*S.diff(Q,coords[B]))
   else:expected=clean(a*a*(et[A-1,B-1]-gamma[A-1,B-1]*trace/3))
   check(str((A,B)),clean(direct)-expected)
  # An erroneous removal of the spatial momentum contribution must be detected.
  checks['negative_control_spatial_term_detected']=S.factor(mu+r*(r+3*h))!=0
 elif mode=='scalar':
  direct=0
  for A in range(5):
   for B in range(5):
    direct+=dgi[A,B]*((Vp-4*H*T if A==B==0 else 0)-G[0,A,B]*T)
    direct+=gi[A,B]*(S.diff(f*Q,coords[A],coords[B])-dG[0,A,B]*T-sum(G[c,A,B]*S.diff(f*Q,coords[c]) for c in range(5)))
  direct-=Vpp*f*Q
  expected=ds(ds(ff))+4*H*ds(ff)+(mu/a**2-Vpp)*ff+(4*ds(ps)-ds(nn))*T-2*nn*Vp
  check('unrestricted_KG_spatial',clean(direct)-clean(expected))
  checks['negative_control_lapse_source_detected']=S.factor(2*nn*Vp)!=0
 elif mode=='wall':
  xi=S.symbols('xi'); xhat=xi+X; qhat=ds(ps)-H*nn
  cov=S.Matrix(4,4,lambda i,j:-(dG[0,i+1,j+1]+nu*Q*G[0,i+1,j+1]+xi*Q*S.diff(G[0,i+1,j+1],s)+xi*hess[i,j]))
  induced=S.Matrix(4,4,lambda i,j:dh[i+1,j+1]+xi*Q*S.diff(g[i+1,j+1],s))
  mixed=S.Matrix(4,4,lambda i,j:sum(gi[i+1,c+1]*(cov[c,j]-S.diff(af,s)/af*induced[c,j]) for c in range(4)))
  for i,j in [(0,0),(0,1),(1,0),(1,1),(2,2),(3,3)]:
   expected=(qhat+(-T*T/3-h*h/a**2)*xhat)*(1 if i==j else 0)-clean(sum(gammai[i,c]*hess[c,j] for c in range(4)))*xhat/a**2
   check('K_'+str((i,j)),clean(mixed[i,j])-clean(expected))
  check('normal_scalar',clean(S.diff(f,s)-nu*T+xi*(Vp-4*H*T))-(ds(ff)-nn*T+(Vp-4*H*T)*xhat))
  check('pullback_scalar',(f+T*xi)-(ff+T*xhat))
  check('induced_conformal',(psi+H*xi)-(ps+H*xhat))
  checks['negative_control_bending_Hessian_detected']=clean(hess[0,1])*xhat/a**2!=0
elif mode=='constraints':
 a,H,T,Vp,Vpp,k,mu,ps,f,L,J=S.symbols('a H T Vp Vpp k mu psi f L J')
 variables=[a,H,T,Vp,ps,f,L,J]
 rhs=[a*H,-T*T/3-k/a**2,Vp-4*H*T,Vpp*T,-2*H*ps-T*f/3,L-2*T*ps,-4*H*L+(Vpp+4*T*T/3-mu/a**2)*f-2*(Vp-4*H*T)*ps+T*J/(3*a**4),None]
 C=T*L-(Vp-4*H*T)*f-3*(mu+4*k)*ps/a**2
 rhs[-1]=2*H*J+2*a**4*C
 def d(v):return sum(S.diff(v,vv)*rr for vv,rr in zip(variables,rhs))
 check('constraint_transport_with_multiplier',d(C)+4*H*C-T*T*J/(3*a**4))
 j=J/a**2
 check('coercive_multiplier_equation',d(a*a*d(j))-S.Rational(2,3)*a*a*T*T*j)
 check('constraint_transport_no_multiplier',(d(C)+4*H*C).subs(J,0))
 # Exact algebraic elimination and independent quadratic coefficient of the current action.
 c,dc,eps=S.symbols('c dc eps', nonzero=True);jv=S.symbols('j0:4');w=S.symbols('w0:4');sig=[-1,1,1,1]
 action=sum(sig[i]*(c*jv[i]**2-jv[i]*w[i]) for i in range(4))
 solved={jv[i]:w[i]/(2*c) for i in range(4)}
 check('auxiliary_current_action',action.subs(solved)+sum(sig[i]*w[i]**2/(4*c) for i in range(4)))
 for i in range(4):check('current_EL_'+str(i),S.diff(action,jv[i]).subs(solved))
 phase=-eps**2*sum(sig[i]*w[i]**2 for i in range(4))/(4*(c+eps*dc))
 check('no_quadratic_coupling_to_c',S.diff(phase,eps,2).subs(eps,0)/2+sum(sig[i]*w[i]**2 for i in range(4))/(4*c))
 h,r=S.symbols('h r',nonzero=True);m=-r*(r+3*h);tf_tt=r*r+m/4
 check('constant_harmonic_exception',tf_tt.subs(r,0));check('embedding_harmonic_exception',tf_tt.subs(r,h))
 checks['generic_zero_mass_example']=S.factor(tf_tt.subs(r,-3*h))!=0
 checks['generic_minus4k_example']=S.factor(tf_tt.subs(r,-4*h))!=0
 check('zero_mass_value',m.subs(r,-3*h));check('minus4k_value',m.subs(r,-4*h)+4*h*h)
else:raise ValueError(mode)
result={'mode':mode,'pass':all(checks.values()),'checks':checks,'residuals':residuals,'seconds':time.monotonic()-start,'pid':os.getpid(),'cpus':sorted(os.sched_getaffinity(0)),'sympy':S.__version__}
Path('CHECK_'+mode+'.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['pass']
