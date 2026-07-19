# CAVEAT — predictions unreliable (added same day, after #69)

The per-rule fixability predictions in classification.md were computed by
comparing each leg goal against the ENTIRE GT reference trajectory, not the
per-leg segment. #69's tracing (reports/issue69_findings.md) showed the
flagged "better candidates" cluster at trajectory fraction 0.87-1.0 — they
are later legs' own targets, not evidence the resolver picked wrong.
Consequences: A1/A2/A3 predictions were artifacts (0/11 real); D1 was a real
defect but its 2-leg lift prediction did not hold; B1 fed the #67 guard that
regressed integrated main. The class-(c) hard-floor count (7) and the raw
per-leg distance data remain usable; every "achievable ceiling" number in
classification.md is overstated. The authoritative re-derivation is #70's
per-leg-segment probe.
