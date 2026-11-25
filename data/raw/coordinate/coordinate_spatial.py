"""Direct coordinate tensors with a spatially varying scalar probe; Linux only."""
import sympy as S
import sys
assert sys.platform == 'linux'
s,t,x,y,z=S.symbols('s t x y z')
h,r,p=S.symbols('h r p', real=True)
a,H,T,V,Vp,Vpp=S.symbols('a H T V Vp Vpp')
coords=[s,t,x,y,z]; af=S.Function('a')(s)
psi,nu,b,E,f=[S.Function(n)(s) for n in ('psi','nu','b','E','f')]
Q=S.exp(r*t+S.I*p*x)
gamma=S.diag(-1,*([S.exp(2*h*t)]*3)); gammai=gamma.inv()
g=S.diag(1,*[af**2*gamma[i,i] for i in range(4)]); gi=g.inv()
G={}
for A in range(5):
 for B in range(5):
  for C in range(5):
   G[A,B,C]=S.simplify(sum(gi[A,D]*(S.diff(g[D,C],coords[B])+S.diff(g[D,B],coords[C])-S.diff(g[B,C],coords[D]))/2 for D in range(5)))
hess=S.Matrix(4,4,lambda i,j:S.diff(Q,coords[i+1],coords[j+1])-sum(G[c,i+1,j+1]*S.diff(Q,coords[c]) for c in range(1,5)))
dh=S.zeros(5); dh[0,0]=2*nu*Q
for i in range(4):
 dh[0,i+1]=dh[i+1,0]=b*S.diff(Q,coords[i+1])
 for j in range(4):dh[i+1,j+1]=2*af**2*(psi*gamma[i,j]*Q+E*hess[i,j])
dgi=-gi*dh*gi; dG={}
for A in range(5):
 for B in range(5):
  for C in range(5):
   dG[A,B,C]=S.simplify(sum((dgi[A,D]*(S.diff(g[D,C],coords[B])+S.diff(g[D,B],coords[C])-S.diff(g[B,C],coords[D]))+gi[A,D]*(S.diff(dh[D,C],coords[B])+S.diff(dh[D,B],coords[C])-S.diff(dh[B,C],coords[D])))/2 for D in range(5)))
subs={S.diff(af,s,2):a*(H**2-T**2/3-h**2/a**2),S.diff(af,s):a*H,af:a}
def clean(v):
 return S.factor(S.expand(v.subs({t:0,x:0}).subs(subs).doit()).subs(V,6+T*T/2+6*h*h/a**2-6*H*H))
def ds(v):
 return S.diff(v,s)+S.diff(v,a)*a*H+S.diff(v,H)*(-T*T/3-h*h/a**2)+S.diff(v,T)*(Vp-4*H*T)+S.diff(v,Vp)*Vpp*T
X=clean(b-af**2*S.diff(E,s)); ps=clean(psi-H*X); nn=clean(nu-ds(X)); ff=clean(f-T*X)
mu=-r*(r+3*h)-p*p
