"""Explicit weighted contraction check for the exact original anchor.

New adapter using unchanged historical flow, Jacobian and interval routines.
No inherited existence premise is used in the self-map/contraction test.
"""
import json
from pathlib import Path
from fractions import Fraction
from flint import arb, ctx, fmpq
import coupled_v002 as C

cfg = json.loads(Path('CONFIG_coupled_v002.json').read_text())
ctx.prec = cfg['precision_bits']; ctx.threads = 1
C.integrator.TAYLOR_ORDER = cfg['taylor_order']
lo, hi = C.minimum, C.maximum
pack = C.integrator.interval_record
ab = lambda x: max(abs(lo(x)), abs(hi(x)))
A = C.integrator.ball(*map(fmpq, cfg['old_anchor_A']))
D = C.integrator.ball(*map(fmpq, cfg['old_anchor_D']))
history = []
for n in range(cfg['anchor_replay_refinements']):
    a, d = C.midpoint(A), C.midpoint(D)
    yc = C.original_flow(arb(a), d, cfg['anchor_steps'])
    yx = C.original_flow(A, lo(D), cfg['anchor_steps'], C.width(D))
    F = C.residual(yc, arb(C.model.LB), arb(C.model.V1))
    J = C.jacobian(yx, arb(C.model.STAR), arb(C.model.V1))
    j00, j01 = map(C.midpoint, J[0]); j10, j11 = map(C.midpoint, J[1])
    det = j00*j11-j01*j10
    assert det != 0
    Y = [[j11/det, -j01/det], [-j10/det, j00/det]]
    E = [[arb(int(i==j))-sum(Y[i][k]*J[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    radii = [(hi(x)-lo(x))/2 for x in (A,D)]
    rows = [sum(ab(e)*r for e,r in zip(row,radii))/ri for row,ri in zip(E,radii)]
    assert max(rows) < 1, ('weighted contraction', rows)
    shift = C.matvec(Y,F); remainder = C.matvec(E,[A-a,D-d])
    N = [x-u+v for x,u,v in zip((a,d),shift,remainder)]
    assert all(lo(old)<lo(new)<=hi(new)<hi(old) for old,new in zip((A,D),N))
    history.append({'iteration':n,'weighted_row_bounds':[str(x) for x in rows],
                    'domain':[pack(A),pack(D)],'image':[pack(x) for x in N]})
    A,D = [C.intersect(old,new) for old,new in zip((A,D),N)]
B = C.original_flow(A,lo(D),cfg['anchor_steps'],C.width(D))[2]
# The historical refinement used these same exact operations. Bind the centers
# consumed by both arithmetic routes, rather than accepting an overlap alone.
candidate = json.loads(Path('candidate.json').read_text())['input_source']
bindings = {}
for name,value in [('C_a',A),('C_b',B)]:
    saved = list(map(Fraction,candidate[name]))
    got = [Fraction(str(lo(value))),Fraction(str(hi(value)))]
    bindings[name] = saved[0] <= got[0] <= got[1] <= saved[1]
assert all(bindings.values()), bindings
out={'pass':True,'precision_bits':ctx.prec,'taylor_order':C.integrator.TAYLOR_ORDER,
     'history':history,'C_a':pack(A),'C_b':pack(B),'contained_in_theorem_inputs':bindings,
     'scope':'Existence and uniqueness in the original rational box by weighted contraction; exact center binding. Same Arb backend, not independent arithmetic.'}
Path('ANCHOR_CHECK.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'pass':True,'row_bounds':[[float(fmpq(x)) for x in h['weighted_row_bounds']] for h in history],'bindings':bindings}))
