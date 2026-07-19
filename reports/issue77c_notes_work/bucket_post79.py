"""Issue #77c Phase 1a: rebuild the per-leg miss bucket table from
reports/gt_battery_main_post79/gt_battery_results.json, using the same
methodology as conversion_probe.md / conversion_probe_v2/probe.md:

  reached_in_order       -- leg_outcomes[i].reached_in_order is True
  cascade_victim         -- reached (some driven pose within tol of goal)
                             but reached_in_order is False
  arrival_blocked        -- never reached (dist_driven > tol) but the
                             resolved goal itself is within tol of the GT
                             reference path (dist_gt <= tol)
  structurally_unreachable -- never reached, and dist_gt > tol too

No rerun -- pure post-hoc classification of the already-committed battery
JSON's leg_outcomes/leg_probe arrays.
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "reports" / "gt_battery_main_post79" / "gt_battery_results.json"
OUT = Path(__file__).parent / "legs_post79.json"

d = json.load(open(SRC))
rows = [r for r in d["scores"] if r["qtype"] == "instruction_following"]

legs = []
for r in rows:
    scene = r["scene"]
    qtext = r["question"]
    outcomes = r["leg_outcomes"]
    probes = {p["i"]: p for p in r["leg_probe"]}
    for lo in outcomes:
        i = lo["i"]
        p = probes.get(i, {})
        tol = lo["tol_used"]
        dist_driven = p.get("min_dist_driven_to_goal_m")
        dist_gt = p.get("dist_goal_to_gt_traj_m")
        reached_in_order = lo["reached_in_order"]
        reached = (dist_driven is not None) and (dist_driven <= tol)
        if reached_in_order:
            bucket = "reached_in_order"
        elif reached:
            bucket = "cascade_victim"
        elif dist_gt is not None and dist_gt <= tol:
            bucket = "arrival_blocked"
        else:
            bucket = "structurally_unreachable"
        excess = (dist_driven - tol) if dist_driven is not None else None
        legs.append({
            "scene": scene, "qtext": qtext[:70], "leg": i, "kind": lo["kind"],
            "bucket": bucket, "tol": tol, "dist_driven": dist_driven,
            "dist_gt": dist_gt, "excess": round(excess, 4) if excess is not None else None,
            "threaded": lo.get("threaded"), "pass_by": lo.get("pass_by"),
        })

counts = {}
for l in legs:
    counts[l["bucket"]] = counts.get(l["bucket"], 0) + 1

print(f"Total legs: {len(legs)}")
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}")

# sanity: recompute mean_ordered_leg_credit (macro-average per question)
by_q = {}
for r in rows:
    key = (r["scene"], r["question"])
    n_legs = r["n_legs"]
    n_ok = sum(1 for lo in r["leg_outcomes"] if lo["reached_in_order"])
    by_q[key] = n_ok / n_legs
macro = sum(by_q.values()) / len(by_q)
print(f"recomputed mean_ordered_leg_credit = {macro:.4f} (reported: {d['aggregate']['instruction_following']['mean_ordered_leg_credit']})")

json.dump(legs, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")

print()
print("-- non-in-order legs, sorted by excess ascending --")
non_io = [l for l in legs if l["bucket"] != "reached_in_order" and l["excess"] is not None]
for l in sorted(non_io, key=lambda l: l["excess"]):
    print(f"{l['scene']:20s} leg{l['leg']} {l['kind']:10s} {l['bucket']:26s} excess={l['excess']:.3f} tol={l['tol']:.3f} q={l['qtext']}")
