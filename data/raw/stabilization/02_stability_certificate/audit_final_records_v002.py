"""Read-only scientific evidence checks using native Python exact rationals.

No interval library, solver imports, floating thresholds, or use of display fields.
This verifies native output identity and the stated logical acceptance predicates;
it does not reconstruct interval propagation.
"""
import hashlib
import json
import sys
from fractions import Fraction as Q
from pathlib import Path

assert sys.platform == "linux"
root = Path.cwd().parent
C = json.loads(Path("CONFIG_vev_coexistence_v001.json").read_text())
F = json.loads(Path("CONFIG_vev_fold_finite_v001.json").read_text())
B = json.loads(Path(sys.argv[1]).read_text())
tests = []


def test(label, ok):
    assert ok, label
    tests.append(label)


def get(ref):
    file = root / ref["path"]
    raw = file.read_bytes()
    test("native SHA-256 " + ref["path"], hashlib.sha256(raw).hexdigest() == ref["sha256"])
    return json.loads(raw)


def rng(x):
    a, b = map(Q, x["exact"] if isinstance(x, dict) else x)
    assert a <= b
    return a, b


test("audit source SHA-256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
     == B["source_sha256"])
whole = get(B["cover"])
fold = get(B["fold"])
n = C["cells_each"]
test("full, nonpilot cover", whole["certified"] and whole["full_cover"] and
     whole["all_cases_pass"] and len(whole["rows"]) == 3*(n+2) and
     whole["cells_each"] == n and whole["common_v"] == C["common_v"])
rows = {(r["kind"], r["index"]): r for r in whole["rows"]}
v0, v1 = rng(C["common_v"])
test("nonempty common v window and exact mass test",
     v0 < v1 and Q(whole["stable_gap_test"]) == Q(1, 100000))
for kind in ("stable", "unstable", "fold"):
    a, b = rng(C[kind]["A"])
    h = (b-a)/n
    for i, aa, bb in [(i,a+i*h,a+(i+1)*h) for i in range(n)] + [
            ("left",a,a), ("right",b,b)]:
        row = rows[kind,i]
        tag = f"{kind}/{i}"
        test(tag+" whole A cover", row["passed"] and rng(row["A_exact"]) == (aa,bb))
        d0,d1 = rng(row["D"])
        e0,e1 = rng(row["event_cover_exact"])
        test(tag+" regular unique event and positive metric mode",
             0 < e0 < e1 <= 1 and 0 < d0 <= d1 <= 1 and
             max(d0,e0) <= min(d1,e1) and d1-d0 < Q(C["max_event_width"]) and
             rng(row["g_s"])[1] < 0 and rng(row["k"])[0] > 0 and
             rng(row["T_b"])[1] < 0 < rng(row["a_b"])[0] and
             rng(row["B_b"])[0] > 0 and Q(row["y_floor"]) > Q(C["positive_y_floor"]) and
             Q(row["test_mu"]) == Q(C[kind]["mu_test"]))
        l,u = rng(row["R"])
        test(tag+" decisive spectral predicate",
             l > 0 if kind == "stable" else u < 0 if kind == "unstable" else
             rng(row["R_A"])[1] < 0 < rng(row["positive_fold_factor"])[0] and
             row["fold_identity_overlap"])

test("strict enclosure by both endpoint images",
     rng(rows["stable","right"]["upper_v"])[1] < v0 < v1 <
     rng(rows["stable","left"]["upper_v"])[0] and
     rng(rows["unstable","left"]["upper_v"])[1] < v0 < v1 <
     rng(rows["unstable","right"]["upper_v"])[0])
test("unique fold sign crossing", rng(rows["fold","left"]["R"])[0] > 0 >
     rng(rows["fold","right"]["R"])[1])
a,b = rng(fold["fold_A"])
al,ah = rng(fold["A_band_exact"])
fl,fh = rng(C["fold"]["A"])
v_l,v_h = rng(fold["fold_v"])
vl,vh = rng(fold["absolute_v_interval_exact"])
r = Q(fold["v_radius_exact"])
c0,c1 = rng(fold["sqrt_coeff"])
test("finite fold enclosure, response and absolute action window",
     fold["pass"] and fold["source_sha256"] == B["cover"]["sha256"] == F["source_sha256"] and
     fl < al < a <= b < ah < fh and b-a <= Q(F["A_width_target"]) and
     a-al == ah-b == Q(F["A_radius"]) and v_l <= v_h < vl < vh < v_l+r and
     r > Q("2.15e-8") and Q(".0238") < c0 <= c1 < Q(".0379") and
     rng(fold["cross_norm_bulk"])[0] > 0 and
     rng(fold["positive_y_all_times"])[0] > Q(1,2) and
     rng(fold["normalized_residual_derivative_magnitude"])[0] > 0)
if B.get("direct"):
    d = get(B["direct"])
    fixed = Q(B["direct_v"])
    test("direct fixed v is within common action window",
         d["pass"] and Q(d["fixed_v"]) == fixed and v0 < fixed < v1 and len(d["rows"]) == 2)
    for k in ("stable", "unstable"):
        x = get(B["direct_"+k])
        m0,m1 = rng(x["mass"])
        a0,a1 = rng(x["A"])
        j0,j1 = rng(C[k]["A"])
        test("direct "+k+" strict contraction in both original junction rows",
             Q(x["fixed_v"]) == fixed and j0 < a0 <= a1 < j1 and
             rng(x["k"])[0] > 0 and rng(x["a_b"])[0] > 0 and
             len(x["background_proof"]) >= 2 and all(
                 len(y["defect_plus_offset"]) == 2 and
                 max(map(Q,y["defect_plus_offset"])) < 1 for y in x["background_proof"]))
        test("direct "+k+" physical mass sign and positive mode",
             x["passed"] and x["kind"] == k and Q(x["metric_y_floor"]) > 0 and
             (m0 > 0 if k == "stable" else m1 < 0))
    ind = get(B["dyadic"])
    test("backend-independent fixed exact action", Q(ind["v"]) == fixed and
         ind["bits"] == 192 and ind["status"] ==
         "PASS_INDEPENDENT_DYADIC_FIXED_ACTION_PAIR_AND_LOWEST_MASSES")
    for r in ind["rows"]:
        kind = r["kind"]
        x = get(B["direct_"+kind])
        test(kind+" integer self-map and junction residues",
             Q(r["v"]) == fixed and len(r["self_map"]) == 2 and
             0 <= min(map(Q,r["self_map"])) <= max(map(Q,r["self_map"])) < 1 and
             len(r["both_rows"]) == 2 and all(rng(f)[0] <= 0 <= rng(f)[1] for f in r["both_rows"]))
        for k in ("A","D","k"):
            a,b = rng(r[k])
            aa,bb = rng(x[k])
            test(kind+" cross-backend enclosure of "+k, 0 < max(a,aa) <= min(b,bb))
        a,b = rng(r["mass"])
        aa,bb = rng(x["mass"])
        left,right = r["residuals"]
        test(kind+" cross-backend mass and whole-profile positivity",
             a < aa <= bb < b and (a > 0 if kind == "stable" else b < 0) and
             rng(left)[0] > 0 > rng(right)[1] and min(map(Q,r["metric_floors"])) > 0)
    test("exactly one direct challenge for each branch",
         sorted(r["kind"] for r in ind["rows"]) == ["stable","unstable"])
out = {"pass": True, "assertions": tests, "count": len(tests),
       "scope": "Exact native record linkage, full coverage, and decision checks; no interval replay."}
Path("FINAL_RECORD_AUDIT.json.tmp").write_text(json.dumps(out,indent=2)+"\n")
Path("FINAL_RECORD_AUDIT.json.tmp").replace("FINAL_RECORD_AUDIT.json")
print("PASS",len(tests))
