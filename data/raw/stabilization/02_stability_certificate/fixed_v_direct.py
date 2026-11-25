"""Two fixed-v BVPs and distinct physical minimum masses on native Linux."""
import concurrent.futures as CF,hashlib,json,math,multiprocessing as MP,os,sys,time,traceback
from pathlib import Path
from fractions import Fraction
from flint import arb
import vev_coexistence as V
assert sys.platform=="linux"
E=V.E;q=V.q;lo=V.lo;hi=V.hi;pack=V.pack
C=json.loads(Path(sys.argv[2]).read_text());v=q(C["v"])
p,ca,cb,la,lb,v0=V.p,V.ca,V.cb,V.la,V.lb,V.v0
def ab(x):return max(abs(lo(x)),abs(hi(x)))
def overlaps(a,b):return max(lo(a),lo(b))<=min(hi(a),hi(b))
def proposal(AA,DD):
    pp,lal,lbl,va,vu=[float(z) for z in [p,la,lb,v0,v]]
    C_a,C_b=[float((lo(x)+hi(x))/2) for x in [ca,cb]]
    def shoot(A,D):
        T=10*(A-va)+(A-C_a)/4
        H=-(lal+10*(A-va)**2+(A-C_a)**2/4)/6
        k=H*H-1-T*T/12+pp*A*A/12
        assert k>0 and D>0
        x=[0.,H,A,T];h=D/C["proposal_steps"]
        def f(z):
            W,H,P,T=z
            return [H,-T*T/3-k*math.exp(-2*W),T,pp*P-4*H*T]
        for _ in range(C["proposal_steps"]):
            a=f(x);b=f([u+h*w/2 for u,w in zip(x,a)])
            c=f([u+h*w/2 for u,w in zip(x,b)]);d=f([u+h*w for u,w in zip(x,c)])
            x=[u+h*(j+2*k+2*l+m)/6 for u,j,k,l,m in zip(x,a,b,c,d)]
        W,H,P,T=x;U=lbl+10*(P-vu)**2+(P-C_b)**2/4
        return H-U/6,T+10*(P-vu)+(P-C_b)/4
    a,d=float(AA),float(DD)
    for _ in range(10):
        g,h=shoot(a,d)
        if max(abs(g),abs(h))<1e-13:break
        ae,de=1e-7,1e-5
        a1=shoot(a+ae,d);a2=shoot(a-ae,d);b1=shoot(a,d+de);b2=shoot(a,d-de)
        aa,cc=[(x-y)/(2*ae) for x,y in zip(a1,a2)];bb,dd=[(x-y)/(2*de) for x,y in zip(b1,b2)]
        det=aa*dd-bb*cc
        a-=(dd*g-bb*h)/det;d-=(-cc*g+aa*h)/det
    return q(str(a)),q(str(d))
def Kball(A):
    H,T,K,Ha,Ta,Ka=V.base(A)
    c=(lo(A)+hi(A))/2
    K=V.cap(K,V.base(arb(c))[2]+Ka*(A-c));assert lo(K)>0
    return H,T,K,Ha,Ta,Ka

def flow(A,D,mu=None):
    H,T,K,Ha,Ta,Ka=Kball(A)
    if mu is None:
        z=[arb(0),H,A,T,arb(0),Ha,arb(1),arb(Ta),arb(0)]
        def rhs(x):
            W,H,P,T,w,hh,pp,tt,U=x;em=(-2*W).exp()
            return [H,-T*T/3-K*em,T,p*P-4*H*T,hh,-2*T*tt/3-(Ka-2*K*w)*em,tt,
                    p*pp-4*(hh*T+H*tt),-4*H*U+(Ka-2*K*w)*em]
    else:
        G=V.gamma(mu,arb(1));den=T*G/2-(p*A-4*H*T);assert hi(den)<0
        f=3*(mu+4*K)/den;z=[arb(0),H,A,T,arb(1),f,G*f/2]
        def rhs(x):
            W,H,P,T,psi,f,L=x;em=(-2*W).exp();Tp=p*P-4*H*T
            return [H,-T*T/3-K*em,T,Tp,-2*H*psi-T*f/3,L-2*T*psi,-4*H*L+(p+4*T*T/3-mu*em)*f-2*Tp*psi]
    E.rhs=rhs
    dl,dh=E.bounds(D);n=C["steps"];s=dh/n
    out=None;cursor=dl;y_floor=q(2)
    for j in range(n):
        b,pic=E.picard_box(z,s);W,H,P,T=b[:4];em=(-2*W).exp();tp=p*P-4*H*T
        assert (H*H-1-T*T/12+p*P*P/12-K*em).contains(0)
        assert hi(T)<-q(".017") and lo(T)>-q(".55") and lo(W.exp())>q(".38")
        if mu is None:
            w,hh,pp,tt,U=b[4:]
            assert (2*H*hh-T*tt/6+p*P*pp/6-(Ka-2*K*w)*em).contains(0)
            assert (U-hh-T*pp/3).contains(0)
        else:
            psi,f,L=b[4:];assert (T*L-tp*f-3*(mu+4*K)*em*psi).contains(0)
            assert lo(2-mu*em)>0 and lo(mu+4*K)>0
            y_floor=min(y_floor,lo((2*W).exp()*psi))
        cell=V.Cell(j*s,s,z,b)
        l=max(dl,j*s);u=min(dh,(j+1)*s)
        if l<=u:
            assert l<=cursor;cursor=max(cursor,u)
            x=cell.eval(l,u)
            out=x if out is None else [E.hull(a,b) for a,b in zip(out,x)]
        z=cell.eval((j+1)*s,(j+1)*s)
    assert out is not None and cursor==dh
    W,H,P,T=out[:4];em=(-2*W).exp()
    if mu is not None:
        psi,f,L=out[4:];G=V.gamma(mu,em)
        return L+G*f/2,y_floor
    w,hh,pp,tt,U=out[4:]
    g=H-(lb+10*(P-v)*(P-v)+(P-cb)*(P-cb)/4)/6
    h=T+10*(P-v)+(P-cb)/4
    J=[[U-h*pp/3,-K*em-T*h/3],[tt+q("41/4")*pp,p*P-4*H*T+q("41/4")*T]]
    return [g,h],J,K,out

def certify(kind):
    try:
        a,d=proposal(*map(q,C["centers"][kind]));ra,rd=map(q,[C["A_radius"],C["D_radius"]])
        src_a,src_b=map(q,V.C[kind]["A"]);assert src_a<a-ra<a+ra<src_b
        history=[]
        for _ in range(C["contraction_rounds"]):
            Ab,Db=E.ball(a-ra,a+ra),E.ball(d-rd,d+rd)
            _,J,K,z=flow(Ab,Db)
            Y=[[x*r for x,r in zip(row,[ra,rd])] for row in J]
            j=[[(lo(y)+hi(y))/2 for y in row] for row in Y]
            det=j[0][0]*j[1][1]-j[0][1]*j[1][0];assert det!=0
            Cinv=[[j[1][1]/det,-j[0][1]/det],[-j[1][0]/det,j[0][0]/det]]
            F,_,_,_=flow(arb(a),arb(d))
            off=[-sum((x*f for x,f in zip(row,F)),arb(0)) for row in Cinv]
            M=[[arb(int(i==k))-sum((Cinv[i][t]*Y[t][k] for t in range(2)),arb(0)) for k in range(2)] for i in range(2)]
            rad=[sum(map(ab,row)) for row in M];test=[ab(x)+y for x,y in zip(off,rad)]
            assert max(test)<1,("strict 2D self map",list(map(float,test)))
            # Root inclusion from the normalized Krawczyk image.
            ai,di=[E.ball(lo(x)-y,hi(x)+y) for x,y in zip(off,rad)]
            Ab=V.cap(Ab,arb(a)+ra*ai);Db=V.cap(Db,arb(d)+rd*di)
            a,d=[(lo(w)+hi(w))/2 for w in [Ab,Db]]
            ra,rd=[(hi(w)-lo(w))/2 for w in [Ab,Db]];assert ra>0 and rd>0
            history.append({"A":pack(Ab),"D":pack(Db),"defect_plus_offset":list(map(str,test))})
        F,J,K,z=flow(Ab,Db);assert all(x.contains(0) for x in F)
        assert src_a<lo(Ab)<=hi(Ab)<src_b and q(".00028")<lo(K)<hi(K)<q(".00032")
        assert lo(1-(Ab-ca)/1000)>0 and lo(1-(z[2]-cb)/1000)>0
        out={"kind":kind,"A":pack(Ab),"D":pack(Db),"k":pack(K),"a_b":pack(z[0].exp()),
             "fixed_v":str(v),"background_proof":history}
        V.atomic("BVP_"+kind+".json",out)
        ml,mh=map(q,C["mass_brackets"][kind]);x,_=flow(Ab,Db,arb(ml));y,_=flow(Ab,Db,arb(mh))
        assert lo(x)>0>hi(y),("strict direct mass signs",pack(x),pack(y))
        hist=[{"mu":str(ml),"residual":pack(x)},{"mu":str(mh),"residual":pack(y)}]
        for _ in range(16):
            if mh-ml<=q(C["mass_width"]):break
            m=(ml+mh)/2;r,_=flow(Ab,Db,arb(m));hist.append({"mu":str(m),"residual":pack(r)})
            if lo(r)>0:ml=m
            elif hi(r)<0:mh=m
            else:break
        V.atomic("MASS_"+kind+".json",hist)
        assert mh-ml<=q(C["mass_width"])
        _,floor=flow(Ab,Db,E.ball(ml,mh));assert floor>0
        out.update(passed=True,mass={"exact":[str(ml),str(mh)],"display":[float(ml),float(mh)]},metric_y_floor=str(floor))
    except BaseException as e:out={"kind":kind,"passed":False,"error":repr(e),"traceback":traceback.format_exc()}
    V.atomic("CASE_"+kind+".json",out)
    return out
if __name__=="__main__":
    with CF.ProcessPoolExecutor(max_workers=2,mp_context=MP.get_context("fork"),initializer=V.pin) as ex:
        rows=list(ex.map(certify,["stable","unstable"]))
    ok=all(r["passed"] for r in rows)
    out={"pass":ok,"fixed_v":str(v),"rows":rows,"scope":"Independent both-junction contraction and original physical mass shoots for one exact common action."}
    V.atomic("FIXED_V_DIRECT.json",out)
    print(json.dumps({"pass":ok,"summary":[{k:r[k] for k in ("kind","error") if k in r}|{k:r[k]["display"] for k in ("A","D","k","mass") if k in r} for r in rows]}))
    assert ok
