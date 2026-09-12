# Joint pose and bounded muscle-support search

The previous three poses were chosen for geometry, not support. To test whether
pose choice alone could resolve the capacity problem, `search_supported_stance.py`
jointly varies root height, 42 joint coordinates, bounded named motor-pool inputs,
and ideal foot-contact forces. It keeps the body, muscle strengths, passive
forces, collision masks and recruitment unchanged. It does not train a neural
model, optimize a movement trajectory, or install a controller.

**Result: no accepted pose.** All three local searches finished within the
120-evaluation bound. None satisfied the declared equilibrium and geometric
tolerances; no candidate was installed in the embodied runner.

| Starting pose | Evaluations | Maximum force-balance error | Maximum penetration |
| --- | ---: | ---: | ---: |
| 0 | 64 | 3.35225 | 0.00007897 |
| 1 | 114 | 5.72207 | 0.00148043 |
| 2 | 120 | 6.93254 | 0.00145651 |

Values use model units; balance includes both force and torque coordinates.
The solver's stopping status is not acceptance. Thresholds were fixed at
balance error <1e-5, penetration <1e-5, and foot-height error <1e-4. The objective
penalizes force balance, contact penetration and foot height, with no preference
for a behavioral trajectory. Roots remain upright in the offline search.

The constraints retain declared joint ranges even where the source does not
enforce them at runtime. Each distal tarsus has one ideal point contact, four
friction rays, and ray weights bounded to twice total body weight. No torso or
self-contact force can help balance gravity. This bounded local search is not a
global infeasibility proof; other poses, contact assumptions or initializations
could yield a different result. It does not justify a claim that the model's CNS
cannot control a fly.

## Recruitment and evidence

Identical motor-recruitment columns are combined into 62 group inputs. This is
an exact reduction for independent normalized inputs in [0,1], not an added
muscle synergy: each group's input is the mean of its interchangeable neurons.
The recorded grouping matches recruitment in the executed full-CNS trial.
One hundred deterministic random-input comparisons agree within 2.23e-16, and
all saved activations reconstruct from saved bounded group inputs.

The complete receipt is `evidence/supported-stance-search-001.json`, including
poses, residuals, muscle activations, ideal contact forces, hashes and limits.
The three searches took 55.30 wall seconds on Vast. No CNS or physical trajectory
ran during optimization. `test_supported_stance.py` verifies recruitment
compression and saved activation reconstruction, not mechanical feasibility.

This attempt provides no supported initial condition to promote. Avoid repeated
unbounded posture searches; muscle-interface calibration and independently
grounded neural response tests remain the more useful next decisions.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python search_supported_stance.py \
  --xml runs/six-leg-body/six_leg.xml --poses runs/refined-poses.json \
  --motor motor-annotations.json --output runs/supported-stance-search.json \
  --max-nfev 120
```
