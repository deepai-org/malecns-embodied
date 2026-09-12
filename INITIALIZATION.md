# Six-leg initialization: collision-confounded

Update: [offline pose refinement](STANCE_SEARCH.md) removes gross initial overlap
without modifying geometry or disabling collision. Bounded muscle support at
those poses still fails the ideal-contact static test. The original initialization
and its evidence below remain unchanged; the CNS trial has not been rerun from
the new poses.

The corrected single-floor body still has an invalidating initialization issue
for interpreting its motion: copied leg assemblies intersect before touching
the ground. This is distinct from the already repaired duplicate floor.

`audit_six_leg_initialization.py` authenticates the body and original muscle
asset, then runs four passive, 50-ms comparisons. Each starts at the unchanged
keyframe with zero muscle commands and zero initial activation. Gravity is
either original or zero; collision is either original or ground-only. The
last setting changes masks in memory only, not the published body or CNS runner.

The derived body begins with 27 active inter-leg contacts, with a maximum
reported penetration of 0.152218 model length units (nominally mm). The original
foreleg asset itself has two thorax/coxa contacts, penetrating 0.0461 and 0.0511;
it is not a collision-free reference. These source contacts are included in the
receipt, not silently ignored. The derived body's root begins at height 4,
with no initial floor contact: this is a drop, not an initialized standing test.

| Gravity | Maximum final joint difference, self-collision on versus off |
| --- | ---: |
| Original | 0.402565 rad |
| Zero | 0.353831 rad |

All four runs complete with finite states. At zero gravity, even the
ground-only case moves by as much as 0.422218 rad from its initial joint pose:
passive muscle forces and minimum activation remain. Thus intersections are a
material confound, **not an established sole cause of collapse**. At 50 ms the
original-collision case still has penetration of 0.138173 with gravity and
0.097213 without it. Merely waiting briefly does not resolve the overlap.

The contact-mask ablation retains floor/body collision; all 76 original geom
type/affinity pairs were checked as `(1, 1)`, and there are no explicit collision
pairs. It removes all body/body interactions, not just the intersecting legs.
The comparisons therefore do not isolate a particular anatomical collision.
Ground-only masks are not proposed as the anatomical repair.

## Consequence for the route

Before interpreting CNS-driven collapse or testing gravity support, establish a
collision-consistent pose and mounting arrangement with feet at the floor and
the torso clear of it. Check initial intersections explicitly. Do not add a
runtime posture controller, disable collision as a hidden fix, or tune neural
gains against this malformed start. If a suitable pose cannot be obtained with
the replicated assemblies, reconsider their geometry rather than elaborate
the neural model around them.

Physical checkpoint replay remains valid: it reproduces the implemented model,
including its flaws. It does not establish biological validity. The previous
full-CNS trial remains evidence of execution and feedback sensitivity, not
standing, useful behavior, or a clean test of CNS competence.

Evidence: `evidence/six-leg-initialization-001.json` contains source and derived
contacts, initial accelerations, 5-ms pose/velocity/contact samples, versions,
input hashes and the executed script hash. No neural simulation runs here.

```sh
.venv/bin/python audit_six_leg_initialization.py \
  --xml runs/six-leg-body/six_leg.xml \
  --body-receipt runs/six-leg-body/receipt.json \
  --source-xml ../FlyMimic/flymimic/assets/models/best_combined_cvt3.xml \
  --output runs/six-leg-initialization.json
```
