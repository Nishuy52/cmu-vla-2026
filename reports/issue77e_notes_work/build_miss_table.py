import json

d = json.load(open('reports/gt_battery_main_post77c/gt_battery_results.json'))
rows = [r for r in d['scores'] if r.get('ordered_leg_credit') is not None]

# (scene, question-substring, leg-index) -> reason
STRUCTURAL = [
    ('home_building_1', 'kettle', 0, 'kettle-q goto leg0'),
    ('home_building_1', 'kettle', 1, 'kettle-q goto leg1'),
    ('livingroom_3', 'pillow farthest', 1, 'frame-unfittable'),
    ('livingroom_3', 'near the stool', 2, 'frame-unfittable'),
    ('arabic_room', 'hookah', 0, 'threading-root leg0 (pocket-clamp)'),
    ('home_building_1', 'nightstand', 0, 'threading-root leg0 (wall-unavailable)'),
    ('livingroom_1', 'lamp closest to the black chair', 0, 'threading-root leg0 (wall-unavailable, 0.039m)'),
]
DEFERRED = [
    ('home_building_2', 'Take the path between the sofa', 1, 'joint corridor/goto trade-off'),
]
# corridor_between threaded=False legs are a separate failure mode (threading violations,
# tracked by n_threading_violations) -- not part of the arrival-blocked pool at all.

misses = []
for r in rows:
    scene = r['scene']
    q = r['question']
    for leg, probe in zip(r['leg_outcomes'], r['leg_probe']):
        i = leg['i']
        if leg['reached_in_order']:
            continue
        if leg['kind'] == 'corridor_between':
            continue  # threading-violation category, not arrival-blocked
        excess = round(probe['min_dist_driven_to_goal_m'] - leg['tol_used'], 4)
        excl = None
        for s, qsub, li, reason in STRUCTURAL:
            if scene == s and (qsub is None or qsub in q) and i == li:
                excl = 'structural: ' + reason
        for s, qsub, li, reason in DEFERRED:
            if scene == s and (qsub is None or qsub in q) and i == li:
                excl = 'deferred: ' + reason
        misses.append({
            'scene': scene, 'question': q, 'leg': i, 'kind': leg['kind'],
            'excess': excess, 'exclude': excl,
        })

pool = [m for m in misses if m['exclude'] is None]
excluded = [m for m in misses if m['exclude'] is not None]
print(f"total goto/via_near arrival misses: {len(misses)}  excluded: {len(excluded)}  fixable pool: {len(pool)}")
print()
print("=== FIXABLE POOL (sorted by excess) ===")
for m in sorted(pool, key=lambda x: x['excess']):
    print(f"{m['excess']:8.4f}  {m['scene']:16s} leg{m['leg']} {m['kind']:9s} {m['question'][:65]}")
print()
print("=== EXCLUDED (structural/deferred) ===")
for m in sorted(excluded, key=lambda x: x['excess']):
    print(f"{m['excess']:8.4f}  {m['scene']:16s} leg{m['leg']} {m['exclude']:55s} {m['question'][:45]}")
