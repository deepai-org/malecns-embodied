# Offline initial-stance search

An initial bounded pose search changes only root height and the 42 leg joint
coordinates. It retains the published body, collision masks and all physical
parameters. It is not a runtime controller and does not test muscle support or
behavior. Three starts use root heights 1.3, 1.7 and 2.1 model units. The objective
penalizes engine-reported penetration and displacement of each distal tarsal
mesh bottom from the floor, with weak pose/foot-placement regularization.

`six-leg-stance-search-001` completes all three 120-evaluation searches. The
best maximum penetration decreases from the original 0.152218 to 0.000063826
model units, with all six foot-height errors below 0.000031. However, none meets
the declared geometric threshold (penetration <1e-5, foot error <1e-4). No
candidate has been installed in the embodied runner.

The first script attempt stopped before optimization because it incorrectly
assumed all source joint ranges were runtime-enforced. In fact only coxa roll
and tibia pitch are limited on each leg. The corrected search respects *all*
declared ranges as conservative offline search bounds, without changing physics.
The receipt records those ranges and runtime limit flags. No result file was
created by that failed attempt.

Geometric feasibility, if found, would only establish a better initial condition.
Static muscle/contact balance, passive dynamics, CNS feedback and rich behavior
remain separate requirements. Failure of a local search does not prove geometric
infeasibility.

Reproduce using the corrected body and the pinned MuJoCo installation:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python search_six_leg_stance.py \
  --xml runs/six-leg-body/six_leg.xml \
  --body-receipt runs/six-leg-body/receipt.json \
  --output runs/six-leg-stance-search.json
```
