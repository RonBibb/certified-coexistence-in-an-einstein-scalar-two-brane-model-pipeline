"""Coordinate linearization of the unrestricted scalar metric, native only."""
import json, os, sys, time
from pathlib import Path
import sympy as S
assert sys.platform == "linux"
started=time.monotonic()
s,t,x,y,z=S.symbols("s t x y z")
h,r=S.symbols("h r", nonzero=True)
a,P,T,H,V,Vp,Vpp=S.symbols("a P T H V Vp Vpp")
coords=[s,t,x,y,z]
af=S.Function("a")(s)
psi,nu,b,E,f=[S.Function(n)(s) for n in ("psi","nu","b","E","f")]
Q=S.exp(r*t)
g=S.diag(1,-af**2,*([af**2*S.exp(2*h*t)]*3))
gi=g.inv()
dh=S.zeros(5)
dh[0,0]=2*nu*Q
dh[0,1]=dh[1,0]=b*r*Q
dh[1,1]=2*af**2*(-psi+E*r*r)*Q
for i in range(2,5):dh[i,i]=2*af**2*S.exp(2*h*t)*(psi-E*h*r)*Q
dgi=-gi*dh*gi
G={}
dG={}
for A in range(5):
 for B in range(5):
  for C in range(5):
   row=[S.diff(g[D,C],coords[B])+S.diff(g[D,B],coords[C])-S.diff(g[B,C],coords[D]) for D in range(5)]
   G[A,B,C]=S.simplify(sum(gi[A,D]*row[D]/2 for D in range(5)))
   dG[A,B,C]=S.simplify(sum((dgi[A,D]*row[D]+gi[A,D]*(S.diff(dh[D,C],coords[B])+S.diff(dh[D,B],coords[C])-S.diff(dh[B,C],coords[D])))/2 for D in range(5)))

subs={S.diff(af,s,2):a*(H**2-T**2/3-h**2/a**2),S.diff(af,s):a*H,af:a}
def clean(v):
 v=S.expand(v.subs(t,0).subs(subs).doit())
 # Hamiltonian background relation removes V.
 return S.factor(v.subs(V,6+T*T/2+6*h*h/a**2-6*H*H))
Ric={}
for A,B in [(0,0),(0,1),(1,1),(2,2)]:
 v=sum(S.diff(dG[C,A,B],coords[C])-S.diff(dG[C,A,C],coords[B]) for C in range(5))
 v+=sum(dG[C,C,D]*G[D,A,B]+G[C,C,D]*dG[D,A,B]-dG[C,B,D]*G[D,A,C]-G[C,B,D]*dG[D,A,C] for C in range(5) for D in range(5))
 v+=(4-S.Rational(2,3)*V)*dh[A,B]-S.Rational(2,3)*Vp*f*Q*g[A,B]
 if A==0:v-=T*S.diff(f*Q,coords[B])
 if B==0:v-=T*S.diff(f*Q,coords[A])
 Ric[A,B]=clean(v)
 # Preserve completed coordinate components.
 Path("COORDINATE_PROGRESS.json").write_text(json.dumps({"component":[A,B],"seconds":time.monotonic()-started})+"\n")

# Expected Einstein residuals in the longitudinal invariant representative.
# Differentiate before replacing background derivatives.
X=b-af**2*S.diff(E,s)
ps=psi-S.diff(af,s)*X/af
nn=nu-S.diff(X,s)
ff=f-T*X
def derivative(v):
 return S.diff(v,s)+S.diff(v,T)*(Vp-4*H*T)+S.diff(v,H)*(-T*T/3-h*h/a**2)+S.diff(v,Vp)*Vpp*T
ps=clean(ps);nn=clean(nn);ff=clean(ff)
def ds(v):
 return derivative(v)+S.diff(v,a)*a*H
mu=-r*(r+3*h)
q=ds(ps)-H*nn
C=T*(ds(ff)-nn*T)-Vp*ff-3*(mu+4*h*h)*ps/a**2-12*H*q
# E_nn=-C, E_st=-3 M r. Tangential formula from independent ADM curvature.
M=q+T*ff/3
iso=((2*mu+6*h*h)*ps+mu*nn)/a**2+3*ds(q)-3*nn*(-T*T/3-h*h/a**2)+12*H*q+T*ds(ff)-nn*T*T+Vp*ff
et=- (2*ps+nn)*(-r*r)/a**2+iso
ex=- (2*ps+nn)*(-h*r)/a**2+iso
trace=-C+et+3*ex
expected={(0,0):-C-trace/3,(0,1):-3*M*r,(1,1):-a*a*(et-trace/3),(2,2):a*a*(ex-trace/3)}
checks={}
for key in Ric:
 difference=clean(Ric[key]-expected[key])
 checks[str(key)]=difference==0
 if difference!=0:Path("RESIDUAL_"+str(key)+".txt").write_text(str(difference))
out={"pass":all(checks.values()),"checks":checks,"seconds":time.monotonic()-started,
 "scope":"Unrestricted homogeneous scalar-coordinate linearization, arbitrary radial profiles and r,h. General-harmonic proof is analytic.",
 "cpus":sorted(os.sched_getaffinity(0))}
Path("UNRESTRICTED_CONTROLS.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
assert out["pass"]
