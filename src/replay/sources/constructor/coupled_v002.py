"""Replay the original exact anchor and validate the coupled upper-only action."""
import json
from pathlib import Path
import platform
import sys

from flint import arb, ctx, fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import exp_bound, gap_checks, maximum, minimum, require
from neighborhood_v001 import jacobian, matvec, midpoint, residual


def width(value):
    return maximum(value)-minimum(value)


def original_flow(A,end,steps,time_span=None):
    model.attach(integrator,model.STAR)
    y=model.launch(A,model.STAR)
    step=end/steps
    for _ in range(steps):
        z,_=integrator.picard_box(y,step)
        require(minimum(z[0].exp())>0,'original warp')
        require(minimum(z[5])>0,'original positive curvature')
        y=integrator.advance(y,z,step)
    if time_span is not None:
        y,_=integrator.picard_box(y,time_span)
    return y


def intersect(a,b):
    return integrator.ball(max(minimum(a),minimum(b)),min(maximum(a),maximum(b)))


def refine_original_anchor(config):
    pack=integrator.interval_record
    A=integrator.ball(*map(fmpq,config['old_anchor_A']))
    D=integrator.ball(*map(fmpq,config['old_anchor_D']))
    steps=config['anchor_steps']
    history=[]
    for _ in range(config['anchor_replay_refinements']):
        a0,d0 = midpoint(A),midpoint(D)
        yc=original_flow(arb(a0),d0,steps)
        yx=original_flow(A,minimum(D),steps,width(D))
        F=residual(yc,arb(model.LB),arb(model.V1))
        J=jacobian(yx,arb(model.STAR),arb(model.V1))
        a,b=map(midpoint,J[0])
        c,d=map(midpoint,J[1])
        determinant=a*d-b*c
        require(determinant!=0,'original root inverse')
        C=[[d/determinant,-b/determinant],[-c/determinant,a/determinant]]
        correction=matvec(C,F)
        E=[[arb(int(i==j))-sum(C[i][l]*J[l][j] for l in range(2))
            for j in range(2)] for i in range(2)]
        remainder=matvec(E,[A-a0,D-d0])
        N=[x-u+v for x,u,v in zip((a0,d0),correction,remainder,strict=True)]
        require(all(minimum(old)<minimum(new) and maximum(new)<maximum(old)
                    for old,new in zip((A,D),N,strict=True)),
                'original anchor strict self-map')
        A,D=[intersect(old,new) for old,new in zip((A,D),N,strict=True)]
        history.append({'A':pack(A),'D':pack(D),'image':[pack(a) for a in N]})
    endpoint=original_flow(A,minimum(D),steps,width(D))
    B=endpoint[2]
    require(minimum(A)>fmpq(99,1000) and maximum(A)<fmpq(1,10),'source C_a bound')
    require(minimum(B)>fmpq(88,1000) and maximum(B)<fmpq(89,1000),'source C_b bound')
    require(width(A)<fmpq(1,10**10) and width(B)<fmpq(1,10**8),'source centre enclosure widths')
    return A,B,{'refinements':history,'C_a':pack(A),'C_b':pack(B)}


def wall_a(A,ca):
    bare = model.lower(A)
    d=A-ca
    tau=bare[0]+d*d/4
    slope=bare[1]+d/2
    return -tau/6,slope/2


def launch(A,p,ca,k_known=None):
    H,T=wall_a(A,ca)
    k=H*H-1-T*T/12+p*A*A/12
    if k_known is not None:
        require(k.contains(arb(k_known)),'exact requested k in lower launch')
        k=arb(k_known)
    require(minimum(k)>0,'positive k launch')
    HA=-T/3
    TA=arb(fmpq(41,4))
    kA=2*H*HA-T*TA/6+p*A/6
    base=[arb(0),H,A,T,arb(0),k]
    sens_A=[arb(0),HA,arb(1),TA,arb(0),kA]
    sens_p=[arb(0),arb(0),arb(0),arb(0),arb(0),A*A/12]
    return base+sens_A+sens_p


def lower_curvature(A,ca):
    H,T=wall_a(A,ca)
    return H*H-1-T*T/12+model.STAR*A*A/12


def coupled_controls(ca,cb):
    checks=gap_checks()['checks']
    require(exp_bound(fmpq(11,10))*fmpq(83,1000)<fmpq(1,4),'coupled initial x barrier')
    require(fmpq(101,100)+(fmpq(1,48)+fmpq(1,10000)*fmpq(16,9))/4
            <fmpq(11,10),'higher-curvature bootstrap')
    ratio=400*(fmpq(25,8)**2*(fmpq(1,5)+fmpq(1,10))+fmpq(3,2))
    require(ratio==fmpq(113400,64) and ratio<1800,'sharpened m/S0 bound')
    require(fmpq(1,1800)-fmpq(4,10000)==fmpq(7,45000),'uniform higher curvature gap')
    ua=(40*model.V0+ca)/41
    ub=(40*model.V1+cb)/41
    la=arb(model.LA)+fmpq(10,41)*(ca-model.V0)**2
    lb=arb(model.LB)+fmpq(10,41)*(cb-model.V1)**2
    d=ua-ub
    delta=-la-lb
    g=fmpq(41,4)
    require(0<minimum(d) and maximum(d)<fmpq(1,100),'original coupled effective center separation')
    require(minimum(ua)>fmpq(9,100) and minimum(ub)>fmpq(9,100),'original coupled positive centers')
    require(fmpq(1,25)<minimum(delta) and maximum(delta)<fmpq(8,125),'original coupled tension difference')
    require(maximum(g*d*d)<minimum((delta-2*g*d*d)/3),'original coupled monotone strict negative trial')
    checks.update(coupled_launch_bootstrap=True,high_curvature_bootstrap=True,
                  coupled_gap_constants=True,original_coupled_obstruction=True)
    return checks


def solve_launch(config,ca):
    lo,hi=map(fmpq,config['new_A_bracket'])
    k0=fmpq(config['new_k'])
    require(maximum(lower_curvature(arb(lo),ca))<k0
            and minimum(lower_curvature(arb(hi),ca))>k0,'coupled root signs')
    initial_box=integrator.ball(lo,hi)
    H,T=wall_a(initial_box,ca)
    derivative=-2*H*T/3-T*fmpq(41,24)+model.STAR*initial_box/6
    require(minimum(derivative)>0,'coupled root monotonicity')
    require(minimum(-H)>fmpq(99,100) and maximum(-H)<fmpq(101,100),'coupled launch H bound')
    require(minimum(-T)>fmpq(1,20) and maximum(-T)<fmpq(83,1000),'coupled launch speed bound')
    for _ in range(config['new_root_bisections']):
        mid=(lo+hi)/2
        F=lower_curvature(arb(mid),ca)-k0
        if maximum(F)<0:
            lo=mid
        elif minimum(F)>0:
            hi=mid
        else:
            # The entire uncertain-centre root family is enclosed using a
            # derivative lower bound and mean-value inequality.
            err=max(abs(minimum(F)),abs(maximum(F)))/minimum(derivative)
            lo=max(lo,mid-err)
            hi=min(hi,mid+err)
            break
    return integrator.ball(lo,hi)


def whole_check(z,p,k_lower,k_upper):
    W,H,Phi,T,_,k=z[:6]
    a=W.exp()
    require(all(q.is_finite() for q in z),'finite coupled path')
    require(minimum(a)>fmpq(3,4),'positive coupled warp')
    require(minimum(H)>fmpq(-11,10) and maximum(H)<fmpq(-9,10),'coupled H box')
    require(0<minimum(Phi) and maximum(Phi)<fmpq(1,10),'coupled scalar box')
    require(minimum(-T)>fmpq(1,20) and maximum(-T)<fmpq(1,4),'coupled speed box')
    require(k_lower<minimum(k) and maximum(k)<k_upper,'coupled positive k box')
    require(model.constraint(z,p).contains(0),'coupled Hamiltonian enclosure')
    require(minimum((-2*H)*(-T)-p*Phi)>0,'monotone shifted bulk weight')
    require(maximum(1-a*a*T*T/(6*k))<0,'coupled E-negative path')


def coupled_flow(A,p,ca,end,steps,k_known=None,time_span=None):
    model.attach(integrator,p)
    y=launch(A,p,ca,k_known)
    step=end/steps
    for _ in range(steps):
        z,_=integrator.picard_box(y,step)
        whole_check(z,p,fmpq(0),fmpq(1,10000))
        y=integrator.advance(y,z,step)
    if time_span is not None:
        y,_=integrator.picard_box(y,time_span)
    whole_check(y,p,fmpq(0),fmpq(1,10000))
    return y


def upper_action(y,cb):
    W,H,Phi,T=y[:4]
    v_new=Phi+T/10+(Phi-cb)/40
    L_new=6*H-10*(Phi-v_new)**2-(Phi-cb)**2/4
    U=L_new+10*(Phi-v_new)**2+(Phi-cb)**2/4
    Up=20*(Phi-v_new)+(Phi-cb)/2
    require((H-U/6).contains(0) and (T+Up/2).contains(0),'coupled upper junction enclosures')
    return {'W_b':W,'a_b':W.exp(),'H_b':H,'Phi_b':Phi,'T_b':T,
            'LB_new':L_new,'v1_new':v_new,'chi_b':(Phi-cb)/2,
            'c_b':1-(Phi-cb)/1000}


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    config=json.loads(Path('CONFIG_coupled_v002.json').read_text())
    ctx.prec=config['precision_bits']
    integrator.TAYLOR_ORDER=config['taylor_order']
    ca,cb,anchor=refine_original_anchor(config)
    checks=coupled_controls(ca,cb)
    A=solve_launch(config,ca)
    end,k0=map(fmpq,(config['new_D'],config['new_k']))
    rec=[]
    for steps in config['new_steps']:
        y=coupled_flow(A,model.STAR,ca,end,steps,k0)
        data=upper_action(y,cb)
        require(minimum(data['c_b'])>0 and minimum(1-(A-ca)/1000)>0,'positive phases')
        require(width(data['LB_new'])<fmpq(1,10**8)
                and width(data['v1_new'])<fmpq(1,10**8),'coupled coefficient widths')
        rec.append((steps,data))
    require(all(rec[0][1][key].overlaps(rec[1][1][key]) for key in rec[0][1]),'coupled refinement agreement')
    pack=integrator.interval_record
    result={
        'status':'PASS_SELECTED_COUPLED_UPPER_ONLY_BACKGROUND_AND_GAP',
        'anchor':anchor,'A_new':pack(A),'k_new':str(k0),'D_new':str(end),
        'chi_a':pack((A-ca)/2),'c_a':pack(1-(A-ca)/1000),
        'checks':checks,'curvature_gap':'8/22500',
        'replications':[{'steps':n,'endpoint':{k:pack(v) for k,v in data.items()}} for n,data in rec],
        'independence':'Two short-flow refinements and fresh original anchor refinement on same Arb backend'
    }
    Path('COUPLED_v002_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
