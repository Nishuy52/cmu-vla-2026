#!/usr/bin/env python3
"""Money table for #83: side-by-side explore-debug JSONL summary."""
import json, sys
for path in sys.argv[1:]:
    rs = []
    for line in open(path):
        try: rs.append(json.loads(line))
        except Exception: pass
    print(f"\n== {path}: {len(rs)} samples")
    for r in rs:
        cm = r.get("costmap", {})
        cands = r.get("frontier_candidates", [])
        acc = sum(1 for c in cands if c.get("accepted"))
        rej = {}
        for c in cands:
            if not c.get("accepted"):
                rej[c.get("reason","?")] = rej.get(c.get("reason","?"), 0) + 1
        print(f"  t={r.get('question_clock_t'):>7} status={r.get('status')} pose={r.get('pose')} "
              f"free={cm.get('free')} unk={cm.get('unknown')} obs={cm.get('obstacle')} "
              f"cands={len(cands)} acc={acc} rej={rej} wp={r.get('chosen_waypoint')}")
