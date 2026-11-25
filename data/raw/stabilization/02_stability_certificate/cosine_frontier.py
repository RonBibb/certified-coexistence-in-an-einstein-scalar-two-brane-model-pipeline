"""Independent native physical metric-pencil shots on exact cosine backgrounds."""
import json,os,sys,hashlib
from pathlib import Path
from fractions import Fraction
from flint import arb,fmpq,ctx
import ivp_engine as E
assert sys.platform=="linux";ctx.prec=768;ctx.threads=1
C=json.loads(Path(sys.argv[1]).read_text())
q=lambda x:fmpq(str(Fraction(str(x))))
lo=lambda a:E.bounds(a)[0];hi=lambda a:E.bounds(a)[1]
def pack(x):
    l,h=E.bounds(x);return {"exact":[str(l),str(h)],"display":[float(l),float(h)]}
def atomic(p,x):
    tmp=Path(p+".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(x,indent=2)+"\n");f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def run(k,mu,mixed):
    ca=arb(q(1)/3)+k;lam=mu+4*k;d=arb(q(2)/3)-lam
    assert lo(ca)>0 and hi(ca)<=q("501/1000") and lo(lam)>0
    # At the exact rational endpoint k=1/6, d=0. Its tiny outward
    # ball can straddle zero without invalidating the Taylor flow.
    if mixed:
        assert lo(1-mu)>0
        G=20+arb(q(1)/4)-arb(q(1)/4)/(1-mu)
    else:G=arb(20)
    assert lo(G)>0
    z=[arb(0),arb(1),arb(1),-2*lam/G]   # z=c*tan(cs), e=sec², y, y'
    def ff(v):
        w,e,y,j=v
        return [ca+w*w,2*w*e,j,d*e*y]
    E.rhs=ff;h=q(1)/C["steps"];floor=q(2);max_pic=0
    for j in range(C["steps"]):
        e,it=E.picard_box(z,h);max_pic=max(it,max_pic)
        assert all(w.is_finite() for w in e)
        assert (e[1]-1-e[0]*e[0]/ca).contains(0)
        assert lo(e[1])>0 and hi(e[1])<q(16)/9
        floor=min(floor,lo(e[2]));z=E.advance(z,e,h)
    w,e,y,j=z;wb=e/(10+w);res=j-lam*wb*y
    return res,{"endpoint":{x:pack(a) for x,a in zip(("warp_gradient_abs","inverse_warp_squared","y","y_prime"),z)},
                "y_floor_exact":str(floor),"y_floor_display":float(floor),"max_picard":max_pic,
                "upper_residual":pack(res),"lower_B":pack(G)}
def bracket(name,fn,br,width):
    a,b=map(q,br);v0,x0=fn(arb(a));v1,x1=fn(arb(b))
    assert lo(v0)>0>hi(v1),(name,"initial signs",pack(v0),pack(v1))
    hist=[{"x":str(a),"eval":x0},{"x":str(b),"eval":x1}]
    for j in range(C["max_bisections"]):
        if b-a<=width:break
        t=(a+b)/2;v,x=fn(arb(t));hist.append({"x":str(t),"eval":x})
        if lo(v)>0:a=t
        elif hi(v)<0:b=t
        else:break
    atomic("HISTORY_"+name+".json",hist)
    assert b-a<=width,(name,"target width")
    _,whole=fn(E.ball(a,b));assert q(whole["y_floor_exact"])>0
    out={"name":name,"exact":[str(a),str(b)],"display":[float(a),float(b)],
         "full_bracket_control":whole}
    atomic("CASE_"+name+".json",out)
    return out
if __name__=="__main__":
    assert q("2/3")*q("4/3")-4 < -3
    assert 1-(q("4/3")+q("16/9"))/20>0
    assert 1-q("2/3")*q("16/9")/10>0
    adverse=q("1/6")-q("1/1000")
    assert q("2/3")-4*adverse>0 and (q("2/3")-4*adverse)*q("16/9")<4*adverse/10
    f=bracket("curvature",lambda k:run(k,arb(0),False),C["k_bracket"],q(C["k_width_target"]))
    a=arb(q(C["mass_at_k"]))
    u=bracket("pure_mass",lambda m:run(a,m,False),C["mass_bracket"],q(C["mass_width_target"]))
    v=bracket("mixed_mass",lambda m:run(a,m,True),C["mass_bracket"],q(C["mass_width_target"]))
    assert q(v["exact"][1])<q(u["exact"][0])
    out={"pass":True,"curvature":f,"pure_mass":u,"mixed_mass":v,"cpus":sorted(os.sched_getaffinity(0)),
         "scope":"Independent exact cosine background metric Sturm; rank-one graph retains curvature critical count; full m positivity and other poles use physical form."}
    atomic("COSINE_FRONTIER.json",out)
    print(json.dumps({"pass":True}|{key:out[key]["display"] for key in ("curvature","pure_mass","mixed_mass")}))
