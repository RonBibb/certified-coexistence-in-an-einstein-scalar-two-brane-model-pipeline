"""Independent integer-interval challenge of the exact fixed-v pair."""
import hashlib
import json
import os
import platform
import sys
from fractions import Fraction as Q
from pathlib import Path

from dyadic_interval import I, BITS
from dyadic_v003_repair3 import P, LA, LB, V0, low, high, pack, intersection
from dyadic_v003_repair3 import launch, curvature, eigen_residual, require


def rhs(z):
    a,b,H,p,T,k,da,db,dH,dp,dT,dk = z
    return [H*a,-H*b,-T.square()/3-k*b.square(),T,P*p-4*H*T,I.exact(0),
            H*da+dH*a,-H*db-dH*b,-2*T*dT/3-dk*b.square()-2*k*b*db,
            dT,P*dp-4*(dH*T+H*dT),I.exact(0)]


def jet(z, order):
    r = [[v] for v in z]
    a,b,H,p,T,k,da,db,dH,dp,dT,dk = r
    def conv(x,y,j):
        return sum((x[n]*y[j-n] for n in range(j+1)), I.exact(0))
    for j in range(order):
        b2, bdb = conv(b,b,j), conv(b,db,j)
        f = [conv(H,a,j),-conv(H,b,j),-conv(T,T,j)/3-k[0]*b2,
             T[j],P*p[j]-4*conv(H,T,j),I.exact(0),
             conv(H,da,j)+conv(dH,a,j),-conv(H,db,j)-conv(dH,b,j),
             -2*conv(T,dT,j)/3-dk[0]*b2-2*k[0]*bdb,
             dT[j],P*dp[j]-4*(conv(dH,T,j)+conv(H,dT,j)),I.exact(0)]
        for x,y in zip(r,f,strict=True):
            x.append(y/(j+1))
    return r


def tube(z,h):
    dt = I.endpoints(0,h)
    y = [u.hull(u+dt*v).pad() for u,v in zip(z,rhs(z),strict=True)]
    for _ in range(36):
        u = [v.hull(v+dt*w) for v,w in zip(z,rhs(y),strict=True)]
        if all(v.inside(w) for v,w in zip(u,y,strict=True)):
            return y
        y = [v.hull(w).pad() for v,w in zip(u,y,strict=True)]
        require(all(-100000 < low(v) and high(v) < 100000 for v in y),
                "integer tangent Picard divergence guard")
    raise RuntimeError("integer tangent Picard tube not validated")


def at(coeff, enclosure, l, r):
    t = I.endpoints(l,r)
    x = []
    for row,z in zip(coeff,enclosure,strict=True):
        q = I.exact(0)
        for c in reversed(row):
            q = c+t*q
        x.append(intersection(q,z))
    return x


def bg_launch(A,ca):
    y = launch(A,ca)
    H,T = y[2],y[4]
    dH,dT = -T/3,I.exact(Q(41,4))
    dk = 2*H*dH-T*dT/6+P*A/6
    Ac = (low(A)+high(A))/2
    y[5] = intersection(y[5],launch(I.exact(Ac),ca)[5]+dk*(A-Ac))
    require(low(y[5])>0,"positive launch curvature")
    return y+[I.exact(0),I.exact(0),dH,I.exact(1),dT,dk]


def solve_flow(A,D,ca,cb,v,steps,order):
    z = bg_launch(A,ca)
    dl,dh = low(D),high(D)
    require(0<dl<=dh<1,"interior upper wall")
    h = dh/steps
    out = None
    for j in range(steps):
        enclosure = tube(z,h)
        a,b,H,p,T,k,da,db,dH,dp,dT,dk = enclosure
        require(low(a)>Q(3,8) and low(b)>0 and high(T)<0 and
                (a*b).contains(1) and (curvature(H,p,T)-k*b.square()).contains(0) and
                (2*H*dH-T*dT/6+P*p*dp/6-dk*b.square()-2*k*b*db).contains(0),
                "regular stationary tangent and constraints")
        lo,hi = j*h,(j+1)*h
        coeff = [x+[y[-1]] for x,y in zip(jet(z,order-1),jet(enclosure,order),strict=True)]
        if dl<=hi:
            q = at(coeff,enclosure,max(dl,lo)-lo,h)
            out = q if out is None else [x.hull(y) for x,y in zip(out,q,strict=True)]
        z = at(coeff,enclosure,h,h)
    require(out is not None,"upper box covered")
    a,b,H,p,T,k,da,db,dH,dp,dT,dk = out
    X = 10*(p-v)+(p-cb)/4
    U = LB+10*(p-v).square()+(p-cb).square()/4
    f = [H-U/6,T+X]
    j = [[dH-X*dp/3,(-T.square()/3-k*b.square())-X*T/3],
         [dT+Q(41,4)*dp,P*p-4*H*T+Q(41,4)*T]]
    return f,j,out


def mid(x):
    return (low(x)+high(x))/2


def ab(x):
    return max(-low(x),high(x))


def matvec(A,x):
    return [sum((u*v for u,v in zip(r,x,strict=True)),I.exact(0)) for r in A]


def contract(a,d,ra,rd,ca,cb,v,n,m):
    A,D = I.endpoints(a-ra,a+ra), I.endpoints(d-rd,d+rd)
    _,J,_ = solve_flow(A,D,ca,cb,v,n,m)
    J = [[x*r for x,r in zip(row,[ra,rd],strict=True)] for row in J]
    j0,j1 = [list(map(mid,row)) for row in J]
    det = j0[0]*j1[1]-j0[1]*j1[0]
    require(det!=0,"exact invertible midpoint preconditioner")
    C = [[j1[1]/det,-j0[1]/det],[-j1[0]/det,j0[0]/det]]
    F,_,_ = solve_flow(I.exact(a),I.exact(d),ca,cb,v,n,m)
    off = matvec(C,[-x for x in F])
    CJ = [matvec(C,[J[0][i],J[1][i]]) for i in range(2)]
    e = [sum(ab(I.exact(int(i==j))-CJ[j][i]) for j in range(2)) for i in range(2)]
    gate = [ab(x)+r for x,r in zip(off,e,strict=True)]
    require(max(gate)<1,"both-junction integer strict self-map "+str(list(map(float,gate))))
    abox,dbox = [I.exact(x)+(z+I.endpoints(-s,s))*r for x,z,s,r in
                 zip([a,d],off,e,[ra,rd],strict=True)]
    return intersection(A,abox),intersection(D,dbox),[str(x) for x in gate]


def atomic(path,value):
    temp = Path(path+".tmp")
    with temp.open("w") as f:
        f.write(json.dumps(value,indent=2)+"\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def main():
    require(sys.platform=="linux" and platform.machine()=="aarch64","native route")
    C = json.loads(Path(sys.argv[1]).read_text())
    for f,h in C["source_sha256"].items():
        require(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,"source hash "+f)
    S = json.loads(Path(C["anchor"]).read_text())
    ca,cb = [I.endpoints(*S[x]) for x in ("C_a","C_b")]
    v = Q(C["v"])
    n,m = C["steps"],C["order"]
    rows = []
    for kind in ("stable","unstable"):
        c = C[kind]
        a,d = map(Q,c["center"])
        ra,rd = map(Q,C["radii"])
        A,D,gates = contract(a,d,ra,rd,ca,cb,v,n,m)
        lower,upper = map(Q,c["branch_A"])
        require(lower<low(A)<=high(A)<upper,"exact branch location")
        F,_,out = solve_flow(A,D,ca,cb,v,n,m)
        require(all(r.contains(0) for r in F),"both original upper rows")
        require(low(1-(A-ca)/1000)>0 and low(1-(out[3]-cb)/1000)>0,
                "actual S6 phase factors")
        row = {"kind":kind,"v":str(v),"A":pack(A),"D":pack(D),"k":pack(out[5]),
               "self_map":gates,"both_rows":[pack(x) for x in F]}
        atomic("BVP_"+kind+".json",row)
        y0 = bg_launch(A,ca)[:6]
        ml,mh = map(Q,c["mass"])
        left,yl,nl = eigen_residual(y0,low(D),high(D),ml,n,m)
        right,yr,nr = eigen_residual(y0,low(D),high(D),mh,n,m)
        require(low(left)>0>high(right) and min(yl,yr)>0,"opposite first mass signs")
        row.update(mass=[str(ml),str(mh)],residuals=[pack(left),pack(right)],
                   metric_floors=[str(yl),str(yr)],mass_picard_splits=[nl,nr])
        atomic("CASE_"+kind+".json",row)
        rows.append(row)
    out = {"status":"PASS_INDEPENDENT_DYADIC_FIXED_ACTION_PAIR_AND_LOWEST_MASSES",
           "v":str(v),"bits":BITS,"rows":rows,"steps":n,"order":m,
           "scope":"Two local solutions of both fixed-action junctions. Integer rounding; common anchor premises and Sturm sign theorem."}
    atomic("FIXED_V_DYADIC.json",out)
    print(out["status"])


if __name__=="__main__":
    main()
