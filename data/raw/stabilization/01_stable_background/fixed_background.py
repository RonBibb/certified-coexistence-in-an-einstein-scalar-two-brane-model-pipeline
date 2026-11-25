"""Generic canonical BG/physical-mu0/parameter-variational RHS adapter.

Validated time integrator algorithms remain in unchanged copied BG engines.
No computation is run or file written on import.
"""
from flint import arb,fmpq
Q=lambda x:fmpq(str(x))
STAR=Q('7385903388887613/18014398509481984')
LA=Q('6755388606266369/1125899906842624');LB=Q('-851862101055633/140737488355328')
V0=Q('3602879701896397/36028797018963968');V1=Q('6520040733916207/72057594037927936')
GAMMA=Q(20);C=Q('9901/100000');SIGMA=Q(100000)
# Optional interval parameter for a separate native action-robustness study.
# Default and every previously frozen run use the nominal exact coefficient.
SEXTIC_BETA=None
def lower(phi,kind='original'):
    d=phi-arb(V0);tau=arb(LA)+10*d*d;tp=20*d;curv=arb(20);curvp=arb(0)
    if kind=='original':return tau,tp,curv,curvp
    alpha=100000*(phi-arb(C))
    if kind=='sextic_closure':
        # Analytic continuation of the stationary tension across z=0, for
        # outward boundary boxes only. Physical requests separately require
        # A<=c; a positive-alpha outer point is not an off-axis equilibrium.
        assert alpha.upper()<arb(Q('1/100')),'analytic closure outer radius'
    else:assert alpha.upper()<0,'off-axis branch requires alpha<0'
    if kind=='quartic':
        eps=arb(Q('1/2'));dd=phi-arb(C)
        return tau-eps*dd*dd/2,tp-eps*dd,curv-eps,curvp
    if kind not in ('sextic','sextic_closure'):raise ValueError(kind)
    beta=arb(50000000) if SEXTIC_BETA is None else SEXTIC_BETA;zeta=arb(5000000000000000)
    z=-2*alpha/(beta+(beta*beta-4*zeta*alpha).sqrt());b=beta+2*zeta*z
    return tau+alpha*z/2+beta*z*z/4+zeta*z*z*z/6,tp+100000*z/2,curv-arb(100000)**2/(2*b),-zeta*arb(100000)**3/(b*b*b)
def make_rhs(p,modes=False):
    n=12 if modes else 6
    def rhs(y):
        W,H,phi,T,Z,k=y[:6];em2=(-2*W).exp();em1=(-W).exp();zero=W*0
        out=[H,-T*T/3-k*em2,T,phi*p-4*H*T,em1,zero]
        if modes:
            for j in (6,9):
                psi,f,g=y[j:j+3]
                out.extend([-2*H*psi-T*f/3,g,-4*H*g+(p+2*T*T)*f+(12*H*T-4*p*phi)*psi])
        for group,dp in enumerate((0,1)):
            q=y[n*(group+1):n*(group+2)];Wq,Hq,phiq,Tq,Zq,kq=q[:6]
            out.extend([Hq,-2*T*Tq/3-kq*em2+2*k*em2*Wq,Tq,
                        p*phiq+dp*phi-4*(Hq*T+H*Tq),-em1*Wq,zero])
            if modes:
                for j in (6,9):
                    psi,f,g=y[j:j+3];psiq,fq,gq=q[j:j+3]
                    out.extend([-2*H*psiq-2*Hq*psi-(Tq*f+T*fq)/3,gq,
                        -4*H*gq-4*Hq*g+(p+2*T*T)*fq+(dp+4*T*Tq)*f
                        +(12*H*T-4*p*phi)*psiq+(12*(Hq*T+H*Tq)-4*(dp*phi+p*phiq))*psi])
        assert len(out)==len(y)
        return out
    return rhs
def launch(A,p,kind='original',modes=False):
    tau,taup,taupp,_=lower(A,kind);H=-tau/6;T=taup/2;k=H*H-1-T*T/12+p*A*A/12
    assert k.lower()>0,'launch k positive'
    base=[arb(0),H,A,T,arb(0),k];groups=[]
    if modes:
        Tp=p*A-4*H*T
        for fa,La in [(arb(1),arb(0)),(arb(0),arb(1))]:
            psi=(T*La-Tp*fa)/(12*k);base.extend([psi,fa,La-2*T*psi])
    for dp in (0,1):
        Ha=arb(0) if dp else -T/3;Ta=arb(0) if dp else taupp/2;Aa=arb(0) if dp else arb(1)
        ka=A*A/12 if dp else 2*H*Ha-T*Ta/6+p*A/6
        q=[arb(0),Ha,Aa,Ta,arb(0),ka]
        if modes:
            Tpq=dp*A+p*Aa-4*(Ha*T+H*Ta)
            for j,(fa,La) in zip((6,9),[(arb(1),arb(0)),(arb(0),arb(1))]):
                psi=base[j];psiq=(Ta*La-Tpq*fa)/(12*k)-psi*ka/k
                q.extend([psiq,arb(0),-2*Ta*psi-2*T*psiq])
        groups.extend(q)
    return base+groups
def attach(E,p,modes=False):
    E.M2_Q=p;rhs=make_rhs(p,modes)
    if hasattr(E,'augmented_rhs'):E.augmented_rhs=rhs
    else:E.rhs=rhs
    return rhs
def quantities(y,p,modes=False):
    n=12 if modes else 6;W,H,phi,T,Z,k=y[:6];rhs=make_rhs(p,modes);v=rhs(y)
    tau=arb(LB)+10*(phi-arb(V1))*(phi-arb(V1));taup=20*(phi-arb(V1))
    G=H-tau/6;h=T+taup/2;Gs=v[1]-taup*T/6;hs=v[3]+10*T
    r={'G':G,'h':h,'G_s':Gs,'H':H,'Phi':phi,'T':T,'warp':W.exp(),'k':k}
    for g,name in enumerate(('A','p')):
        z=y[n*(g+1):n*(g+2)];Gq=z[1]-taup*z[2]/6;hq=z[3]+10*z[2];Dq=-Gq/Gs
        r['D_'+name]=Dq;r['q_'+name]=hq+hs*Dq;r['Phi_'+name]=z[2]+T*Dq
    r['branch_A_p']=-r['q_p']/r['q_A'];r['branch_p_A']=-r['q_A']/r['q_p']
    r['branch_Phi_b_p']=r['Phi_p']+r['Phi_A']*r['branch_A_p']
    if modes:
        Ls=[];Lq=[[],[]];Tp=v[3]
        for j in (6,9):
            psi,f,g=y[j:j+3];L=g+2*T*psi;Ls.append(L)
            Lprime=-4*H*L+(p+4*T*T/3)*f-2*Tp*psi
            for group,name in enumerate(('A','p')):
                z=y[n*(group+1):n*(group+2)];Lpartial=z[j+2]+2*z[3]*psi+2*T*z[j]
                Lq[group].append(Lpartial+Lprime*r['D_'+name])
        r['m11']=y[7];r['m12']=y[10];r['m21']=Ls[0];r['m22']=Ls[1];r['g0']=2*Ls[0]/Ls[1]
        for group,name in enumerate(('A','p')):r['g0_'+name]=2*(Lq[group][0]*Ls[1]-Ls[0]*Lq[group][1])/(Ls[1]*Ls[1])
        r['g0_branch_A']=r['g0_A']+r['g0_p']*r['branch_p_A']
        num=Ls[0]+10*y[7];den=Ls[1]+10*y[10]
        r['Q20']=2*num/den
        for group,name in enumerate(('A','p')):
            z=y[n*(group+1):n*(group+2)];dq=r['D_'+name]
            nq=Lq[group][0]+10*(z[7]+y[8]*dq)
            eq=Lq[group][1]+10*(z[10]+y[11]*dq)
            r['Q20_'+name]=2*(nq*den-num*eq)/(den*den)
        r['Q20_branch_A']=r['Q20_A']+r['Q20_p']*r['branch_p_A']
    return r
def constraint(y,p):
    W,H,phi,T,Z,k=y[:6];return H*H-1-T*T/12+p*phi*phi/12-k*(-2*W).exp()
