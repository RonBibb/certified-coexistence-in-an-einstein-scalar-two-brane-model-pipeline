"""Exact symbolic independent challenges of compensated fold/mass identities."""
import hashlib
import json
from pathlib import Path
import platform
import sys

import sympy as S


def main():
    if sys.platform!='linux' or platform.machine()!='aarch64':
        raise RuntimeError('native route')
    a,H,Phi,T,k,p,w,h,eta,t,K=S.symbols('a H Phi T k p w h eta t K',nonzero=True)
    Tp=p*Phi-4*H*T
    rows={a:a*H,H:-T*T/3-k/a**2,Phi:T,T:Tp,
          w:h,h:-2*T*t/3-K/a**2+2*k*w/a**2,eta:t,t:p*eta-4*h*T-4*H*t}
    derivative=lambda x:sum(S.diff(x,q)*v for q,v in rows.items())
    c=H**2-1-T*T/12+p*Phi*Phi/12-k/a**2
    dc=2*H*h-T*t/6+p*Phi*eta/6-(K-2*k*w)/a**2
    U=h+T*eta/3
    xi=a**2*U/k
    psi=w-K/(2*k)+H*xi
    f=eta+T*xi
    L=t+Tp*xi
    check={}
    def equal(label,left,right=0):
        expression=S.factor(S.together(left-right))
        check[label]=expression==0
        if not check[label]:
            raise AssertionError((label,str(expression)))
    equal('exact bulk first integral derivative',derivative(c))
    equal('linearized first integral derivative',derivative(dc))
    equal('compensated noncancelling U flow',derivative(U)+4*H*U-(K-2*k*w)/a**2,2*dc)
    equal('static lapse and radial shift',derivative(xi)+2*psi,2*a**2*dc/k)
    equal('physical momentum row',derivative(psi)+2*H*psi+T*f/3,2*H*a**2*dc/k)
    equal('physical f derivative',derivative(f)-L+2*T*psi,2*T*a**2*dc/k)
    equal('physical Hamiltonian row',T*L-Tp*f-12*k*psi/a**2,-6*dc)
    equal('physical L evolution',derivative(L)+4*H*L-(p+4*T*T/3)*f+2*Tp*psi,
          2*Tp*a**2*dc/k)

    w2,h2,eta2,t2,K2=S.symbols('w2 h2 eta2 t2 K2')
    variations={a:a*w,H:h,Phi:eta,T:t,k:K,w:w2,h:h2,eta:eta2,t:t2,K:K2}
    var=lambda x:sum(S.diff(x,q)*v for q,v in variations.items())
    equal('second varied curvature force',var(var(k/a**2)),
          (K2-4*K*w-2*k*w2+4*k*w*w)/a**2)
    U2=var(U)
    equal('second compensated force',var(-4*H*U+(K-2*k*w)/a**2),
          -4*H*U2-4*h*U+(K2-4*K*w-2*k*w2+4*k*w*w)/a**2)
    equal('second varied scalar acceleration',var(var(Tp)),p*eta2-4*(h2*T+2*h*t+H*t2))

    v,Cb,Lb=S.symbols('v Cb Lb')
    Ub=Lb+10*(Phi-v)**2+(Phi-Cb)**2/4
    G=H-Ub/6
    hh=T+S.diff(Ub,Phi)/2
    S0=T+(Phi-Cb)/4
    v_match=Phi+S0/10
    equal('stationary scalar row exactly',hh.subs(v,v_match))
    equal('single changed coefficient event',6*G.subs(v,v_match),
          6*H-Lb-(Phi-Cb)**2/4-S0**2/10)
    equal('independent upper vacuum derivative of metric row',S.diff(G,v).subs(v,v_match),-S0/3)
    equal('independent upper vacuum derivative of scalar row',S.diff(hh,v),-10)
    equal('metric time row at actual scalar root',derivative(G).subs(v,v_match),-k/a**2)
    equal('compensated upper derivative at scalar root',var(G).subs(v,v_match),U)
    Gd,Gv,ha,hd=S.symbols('Gd Gv ha hd',nonzero=True)
    va=-(U-Gd*ha/hd)/(Gv+10*Gd/hd)
    da=(10*va-ha)/hd
    equal('two-residual differentiated matching',U+Gd*da+Gv*va)
    equal('physical mismatch proportional to action slope',ha+hd*xi,
          (10-hd*a**2*Gv/k)*va+hd*a**2*(U+(-k/a**2)*da+Gv*va)/k)

    y,F,ym,Fm,Q,W,lam,P=S.symbols('y F ym Fm Q W lam P',nonzero=True)
    dyds=F/P
    dFds=(Q-lam*W)*y
    dyms=Fm/P
    dFms=(Q-lam*W)*ym-W*y
    equal('mass Wronskian derivative',dyds*Fm+y*dFms-dyms*F-ym*dFds,-W*y*y)
    mu=S.symbols('mu')
    Gamma=21-1/(2-mu/a**2)
    equal('finite dynamical oscillator derivative',S.diff(Gamma,mu).subs(mu,0),-1/(4*a**2))

    executable=Path(sys.executable)
    package=Path(S.__file__)
    record={'status':'PASS_EXACT_FOLD_AND_MASS_IDENTITIES','checks':check,
            'runtime':{'python':str(executable),'python_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
                       'sympy':str(package),'sympy_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
                       'sympy_version':S.__version__},
            'scope':'Exact symbolic identities, separate from validated stationary folds and mass data.'}
    Path('FOLD_IDENTITY_CONTROLS_v004_RESULT.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
