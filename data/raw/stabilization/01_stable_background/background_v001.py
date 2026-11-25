"""Nominal frozen-action exact bounds and outward whole-path checks on Linux."""
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import flint
from flint import arb, ctx, fmpq
import bg_independent as integrator
import fixed_background as model


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def maximum(x):
    return x.upper().fmpq()


def minimum(x):
    return x.lower().fmpq()


def launch_k_exact(a):
    t = 10 * (a - model.V0)
    h = -(model.LA + 10 * (a - model.V0)**2) / 6
    return h*h - 1 - t*t/12 + model.STAR*a*a/12


def exp_bound(x, terms=12):
    """Rational strict upper bound by geometric domination of the tail."""
    total = term = fmpq(1)
    for j in range(1, terms + 1):
        term *= x / j
        total += term
    return total + term*x/(terms+1)/(1-x/(terms+2))


def gap_checks():
    results = {}

    def check(label, value):
        require(value, label)
        results[label] = True

    delta = -model.LA-model.LB
    d = model.V0-model.V1
    check('old tension margin', fmpq(1,25) < delta < fmpq(8,125))
    check('old center and separation bounds', 0 < d < fmpq(1,100)
          and min(model.V0, model.V1) > fmpq(9,100))
    check('old monotone trial strict negativity', 10*d*d < 2*(delta-20*d*d)/6)
    check('bootstrap warp barrier', exp_bound(fmpq(11,40)) < fmpq(4,3))
    check('bootstrap scalar speed ceiling', fmpq(81,1000)*exp_bound(fmpq(11,10)) < fmpq(1,4))
    check('bootstrap scalar speed floor', fmpq(18,5)/20-fmpq(21,500)>0)
    check('bootstrap scalar positive', fmpq(23,250)-fmpq(1,4)**2>0)
    k_max = fmpq(1,50_000_000)
    check('bootstrap H floor', fmpq(101,100)+(fmpq(1,48)+k_max*fmpq(16,9))/4 < fmpq(11,10))
    check('fundamental norm bound', exp_bound(fmpq(3,4)) < 3
          and fmpq(11,5)+fmpq(1,48)<3)
    check('kinetic coefficient', (6+fmpq(1,16))/4 < fmpq(7,4))
    check('lower and upper trace coefficients', 10-fmpq(22,5)>5
          and fmpq(18,5)-fmpq(21,500)*20>0)
    weights_m = [fmpq(21,4)*n for n in (360_000,32_400,225)]
    weights_b = [fmpq(3,4)**4*n for n in (5,5,fmpq(1,400))]
    lower = min(b/m for b,m in zip(weights_b,weights_m,strict=True))
    check('uniform shifted gap', lower > fmpq(1,2_000_000))
    check('uniform positive gap', fmpq(1,2_000_000)-4*k_max == fmpq(21,50_000_000))
    check('oscillator gap', fmpq(9,16) > fmpq(21,50_000_000))
    return {'checks': results, 'actual_coefficient_ratio_minimum': str(lower),
            'uniform_claimed_gap': '21/50000000'}


def validate_path(launch, count, end, k0):
    rhs = model.make_rhs(model.STAR)
    y = launch
    whole = y.copy()
    step = end/count
    max_picard = 0
    for _ in range(count):
        box, attempts = integrator.picard_box(y,step)
        max_picard = max(max_picard, attempts)
        W,H,phi,T,_,k = box[:6]
        require(all(value.is_finite() for value in box), 'all whole-cell states finite')
        require(minimum(W.exp())>fmpq(3,4) and maximum(W.exp())<fmpq(101,100), 'whole warp')
        require(minimum(H)>fmpq(-11,10) and maximum(H)<fmpq(-9,10), 'whole H')
        require(minimum(phi)>0 and maximum(phi)<fmpq(1,10), 'whole scalar')
        require(minimum(-T)>fmpq(1,20) and maximum(-T)<fmpq(1,4), 'whole scalar derivative')
        require(k.contains(arb(k0)) and model.constraint(box, model.STAR).contains(0), 'constraint checks')
        require(maximum(1-W.exp()**2*T**2/(6*k))<0, 'strict departure from E>0')
        require(maximum(rhs(box)[1])<0, 'H strictly decreases')
        whole = [integrator.hull(a,b) for a,b in zip(whole,box,strict=True)]
        y = integrator.advance(y,box,step)
    require(model.constraint(y, model.STAR).contains(0), 'endpoint constraint')
    W,H,phi,T,_,_ = y[:6]
    lb, v1 = 6*H-T*T/10, phi+T/10
    require((H-(lb+10*(phi-v1)**2)/6).contains(0), 'upper H enclosure')
    require((T+10*(phi-v1)).contains(0), 'upper T enclosure')
    endpoint = dict(zip(('W','H','Phi','T','Z','k'),y[:6],strict=True))
    endpoint.update(a=W.exp(), LB_new=lb, v1_new=v1)
    return endpoint, {
        'steps': count,
        'max_picard_attempts': max_picard,
        'whole_path': {key: integrator.interval_record(value)
                       for key,value in zip(('W','H','Phi','T','Z','k'),whole[:6],strict=True)},
        'endpoint': {key: integrator.interval_record(value) for key,value in endpoint.items()}
    }


def main():
    require(sys.platform == 'linux' and platform.machine() == 'aarch64', 'native route')
    config = json.loads(Path('CONFIG_background_v001.json').read_text())
    ctx.prec = config['precision_bits']
    integrator.TAYLOR_ORDER = config['taylor_order']
    model.attach(integrator, model.STAR)
    lo,hi = map(fmpq,config['launch_bracket'])
    k0 = fmpq(config['curvature'])
    end = fmpq(config['separation'])
    require(launch_k_exact(lo)<k0<launch_k_exact(hi), 'exact root bracket')
    A = integrator.ball(lo,hi)
    original_tension, slope, _, _ = model.lower(A)
    H = -original_tension/6
    T = slope/2
    derivative = -2*H*T/3-T*10/6+model.STAR*A/6
    require(minimum(derivative)>0, 'unique root derivative')
    require(minimum(-H)>fmpq(99,100) and maximum(-H)<fmpq(101,100), 'lower H')
    require(minimum(-T)>fmpq(69,1000) and maximum(-T)<fmpq(81,1000), 'lower T')
    for _ in range(config['exact_bisections']):
        mid = (lo+hi)/2
        if launch_k_exact(mid)<k0:
            lo=mid
        else:
            hi=mid
    require(launch_k_exact(lo)<k0<=launch_k_exact(hi), 'refined exact signs')
    init = model.launch(integrator.ball(lo,hi),model.STAR)
    require(init[5].contains(arb(k0)), 'known exact curvature enclosable')
    init[5] = arb(k0)
    require(model.constraint(init,model.STAR).contains(0), 'launch constraint')
    first,r1 = validate_path(init,config['steps'][0],end,k0)
    second,r2 = validate_path(init,config['steps'][1],end,k0)
    require(all(first[name].overlaps(second[name]) for name in first), 'refinement overlap')
    max_width = fmpq(config['maximum_action_coefficient_diameter'])
    for name in ('LB_new','v1_new'):
        require(maximum(second[name])-minimum(second[name])<max_width, name+' enclosure size')
    extension = next(Path(flint.__file__).parent.rglob('arb*.so'))
    dependencies = {}
    for path in (Path(flint.__file__), extension, Path('bg_independent.py'),
                 Path('fixed_background.py'), Path('CONFIG_background_v001.json'),
                 Path(config['action']), Path(config['candidate'])):
        dependencies[str(path)] = sha256(path.read_bytes()).hexdigest()
    result = {
        'status':'PASS_UPPER_WALL_ONLY_BACKGROUND_AND_EXACT_GAP_CONTROLS',
        'flint_version':flint.__version__,
        'dependencies_sha256':dependencies,
        'precision_bits':ctx.prec,
        'A_root_bounds':[str(lo),str(hi)],
        'A_display':[float(lo),float(hi)],
        'A_coarse_K_derivative':integrator.interval_record(derivative),
        'constraint_method':'Exact first integral identity plus per-cell containment',
        'original_wall_coefficients':{'LA':str(model.LA),'LB':str(model.LB),
                                      'v0':str(model.V0),'v1':str(model.V1)},
        'replications':[r1,r2],
        'gap_controls':gap_checks(),
        'independence':'Two refinements of one outward implementation with the same Arb backend'
    }
    Path('BACKGROUND_v001_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
