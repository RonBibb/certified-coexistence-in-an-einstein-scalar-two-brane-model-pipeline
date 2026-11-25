"""Uniform parameter-box existence/uniqueness for a finite deformed action."""
import json
from pathlib import Path
import platform
import sys

from flint import arb, ctx, fmpq

import bg_independent as integrator
import fixed_background as model
from background_v001 import maximum, minimum, require


def ball(centre, radius):
    return integrator.ball(centre-radius,centre+radius)


def absolute_upper(x):
    return max(abs(minimum(x)), abs(maximum(x)))


def midpoint(x):
    return (minimum(x)+maximum(x))/2


def box_check(z):
    W,H,phi,T,_,k = z[:6]
    a = W.exp()
    require(minimum(a)>fmpq(3,4), 'whole warp lower')
    require(minimum(H)>fmpq(-11,10) and maximum(H)<fmpq(-9,10), 'whole H')
    require(minimum(phi)>0 and maximum(phi)<fmpq(1,10), 'whole scalar')
    require(minimum(-T)>fmpq(1,20) and maximum(-T)<fmpq(1,4), 'whole scalar speed')
    require(minimum(k)>fmpq(1,200_000_000) and maximum(k)<fmpq(1,50_000_000), 'curvature band')


def propagate(A,p,end,steps,tube_length=0):
    model.attach(integrator,p)
    state=model.launch(A,p)
    step=end/steps
    for _ in range(steps):
        enclosure,_ = integrator.picard_box(state,step)
        box_check(enclosure)
        state=integrator.advance(state,enclosure,step)
    if tube_length:
        state,_ = integrator.picard_box(state,tube_length)
    box_check(state)
    return state


def residual(y,lb,v1):
    _,H,phi,T = y[:4]
    d=phi-v1
    return [H-(lb+10*d*d)/6,T+10*d]


def jacobian(y,p,v1):
    W,H,phi,T,_,k = y[:6]
    _,HA,phiA,TA = y[6:10]
    G_A=HA-(10*(phi-v1)*phiA)/3
    _,h = residual(y,arb(0),v1)
    G_D=-k*(-2*W).exp()-T*h/3
    h_A=TA+10*phiA
    h_D=p*phi-4*H*T+10*T
    return [[G_A,G_D],[h_A,h_D]]


def matvec(matrix, vector):
    return [sum(a*b for a,b in zip(row,vector,strict=True)) for row in matrix]


def main():
    require(sys.platform=='linux' and platform.machine()=='aarch64','native route')
    config = json.loads(Path('CONFIG_neighborhood_v001.json').read_text())
    q = lambda key: fmpq(config[key])
    ctx.prec = config['precision_bits']
    integrator.TAYLOR_ORDER = config['taylor_order']
    x = [q('centre_A'),q('centre_D')]
    radii = [q('radius_A'),q('radius_D')]
    A = ball(x[0],radii[0])
    p = ball(model.STAR,q('radius_p'))
    lb = ball(q('centre_LB'),q('radius_LB'))
    v1 = ball(q('centre_v1'),q('radius_v1'))
    require(fmpq(2,5)<minimum(p) and maximum(p)<fmpq(21,50),'p bootstrap band')
    require(fmpq(1,6)<x[1]-radii[1] and x[1]+radii[1]<fmpq(1,4),'D bootstrap band')
    centre_y = propagate(arb(x[0]),p,x[1],config['steps'])
    domain_y = propagate(A,p,x[1]-radii[1],config['steps'],2*radii[1])
    F0 = residual(centre_y,lb,v1)
    J = jacobian(domain_y,p,v1)
    a,b = map(midpoint,J[0])
    c,d = map(midpoint,J[1])
    det = a*d-b*c
    require(det!=0,'invertible rational midpoint Jacobian')
    C = [[d/det,-b/det],[-c/det,a/det]]
    shift = matvec(C,F0)
    error = []
    for i in range(2):
        error.append([arb(int(i==j))-sum(C[i][l]*J[l][j] for l in range(2))
                      for j in range(2)])
    row_sums = [sum(absolute_upper(e)*r for e,r in zip(row,radii,strict=True))/ri
                for row,ri in zip(error,radii,strict=True)]
    shift_fractions = [absolute_upper(s)/r for s,r in zip(shift,radii,strict=True)]
    totals = [a+b for a,b in zip(row_sums,shift_fractions,strict=True)]
    require(max(row_sums)<1,'strict contraction')
    require(max(totals)<1,'strict self-map for all parameter combinations')
    pack=integrator.interval_record
    result = {
        'status':'PASS_OPEN_THREE_PARAMETER_STABLE_BVP_NEIGHBORHOOD',
        'domains':{'A':pack(A),'p':pack(p),'LB_new':pack(lb),'v1_new':pack(v1),
                   'D':[str(x[1]-radii[1]),str(x[1]+radii[1])]},
        'curvature':pack(domain_y[5]),
        'Jacobian':[[pack(a) for a in row] for row in J],
        'rational_inverse_preconditioner':[[str(a) for a in row] for row in C],
        'midpoint_Jacobian_det':str(det),
        'residual_at_center':[pack(a) for a in F0],
        'weighted_contraction_rows':[str(a) for a in row_sums],
        'weighted_correction':[str(a) for a in shift_fractions],
        'weighted_total_max':str(max(totals)),
        'weighted_total_display':[float(a) for a in totals],
        'gap_bound':'21/50000000',
        'scope':'Independent p, upper tension and upper vacuum perturbations with all other action coefficients fixed. For each fixed parameter triple, exactly one BVP root inside the stated A,D box.'
    }
    Path('NEIGHBORHOOD_v001_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
