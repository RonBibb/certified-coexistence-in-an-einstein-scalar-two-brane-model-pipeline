"""Read-only exact conversion check, using only the Python standard library."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib, json

R=Path(__file__).resolve().parent
record=json.loads((R/'growth_rates.json').read_text())
source=R/'inputs/saved_pair.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
rows=json.loads(source.read_text())['rows']
def bounds(d): return list(map(Q,d.get('exact',d.get('exact_bounds'))))
for raw,rate in zip(rows,record['rows']):
    assert raw['kind']==rate['kind']
    mass,curvature=bounds(raw['mass']),bounds(raw['k'])
    assert mass==list(map(Q,rate['mass_exact']))
    assert curvature==list(map(Q,rate['curvature_exact'])) and curvature[0]>0
    values=[m/k for m in mass for k in curvature]
    xl,xh=min(values),max(values)
    assert [xl,xh]==list(map(Q,rate['mass_ratio_exact']))
    lo,hi=map(Q,rate['slow_exponent_ratio_strict'])
    assert -Q(3,2)<lo<hi
    assert (lo+Q(3,2))**2<Q(9,4)-xh
    assert (hi+Q(3,2))**2>Q(9,4)-xl
    assert hi<0 if rate['kind']=='stable' else 0<lo<hi<1
assert len(rows)==len(record['rows'])==2
print(json.dumps({'pass':True,'cases':2,'source_hash_verified':True,'exact_square_comparisons':4,'scientific_integrations':0}))
