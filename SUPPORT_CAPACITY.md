# Support capacity and recruitment are both unresolved

The three geometrically acceptable poses do not merely need a small neural
gain adjustment. A minimax static calculation minimizes the largest normalized
input required for muscle/contact balance, keeping passive forces unchanged.
It extrapolates active force above input 1 **only algebraically**. No out-of-range
input, gain change or optimizer controller is installed in the simulator.

| Recruitment assumption | Pose 0 | Pose 1 | Pose 2 |
| --- | ---: | ---: | ---: |
| All 90 units independent | 11.58 | 5.65 | 4.70 |
| Existing named pools | Infeasible | 58.95 | Infeasible |
| Only mapped 78 units, independently driven | 52.47 | 43.44 | Infeasible |
| Named pools plus independent drive for the 12 unmapped units | 12.97 | 8.31 | 7.76 |

Entries are minimum peak inputs, where the normal actuator-input ceiling is 1.
Infeasible means infeasible even without that ceiling, within the specific
pose/ideal-contact assumptions. The two additional mappings are diagnostic
relaxations, **not proposed anatomical assignments**. They separate restrictions
from shared recruitment and missing muscle drive. They show why merely increasing
global strength cannot resolve the current mapping at every tested pose.

The static assay retains the qualifications in [STANCE_SEARCH.md](STANCE_SEARCH.md):
one ideal point contact per distal tarsus, friction-limited forces, no body-contact
support, fixed pose and zero velocity. Feasible solutions include balance errors,
input-bound errors, dual objective and dual stationarity checks. These checks
support the numerical optimization, not physiological validity. The minimum
input is not automatically a recommended Fmax multiplier: changing Fmax can
also change passive forces, which this calculation holds fixed.

## Biological force cross-check

[Azevedo et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/)
observed tibia-tip forces approaching 100 micronewtons, approximately ten fly
body weights. The paper also demonstrates differing motor-unit force regimes;
equal mean activity is not a measured recruitment law.

`audit_tibia_force.py` measures the model's static active tibia-flexion torque
at activation 1, subtracts passive actuator torque, and divides by the tibia-tip
rotational lever arm. The tip is the tarsus-1 body origin, not the distal toe.
All other coordinates are ideally held fixed. Dividing that force by the
model's own body weight avoids assuming an SI mass-unit conversion.

- Original source foreleg keyframe: **0.0832 body weights**.
- Replicated body at the three candidate poses: **0.0375–0.0845 body weights**.

This is a substantial warning against treating the imported actuator parameters
as a biologically calibrated whole-fly force model. It is **not** a justified
100-fold scaling prescription: only one modeled flexor unit is activated, the
source muscle inventory is incomplete, and the experimental posture, probe
direction, specimen and recruitment differ. The experimental voluntary peak is
not an isometric maximum at every modeled pose. Geometry/lever arms, coverage
and force parameters need checking together before fitting a correction.

An attempted lookup of `MNml77`/`MNml78` did not establish a promotor assignment.
The [VFB MNml78 record](https://www.virtualflybrain.org/blog/2022/01/01/mnml78-fbbt_20011216/)
calls it an uncharacterized putative type. Unknown named motor neurons therefore
remain unassigned; no correspondence was invented to make support pass.

## Evidence and next decision

`evidence/six-leg-stance-support-003.json` preserves the first minimax assay
(executed script in git `c94c492`). `-004.json` adds the two recruitment
relaxations. `evidence/tibia-force-001.json` contains the force/lever-arm ratios,
input and runner hashes, and biological-reference scope.

The next interface revision should address missing muscle coverage and
physiological force/recruitment calibration, with the same support tests retained.
Do not tune CNS recurrence to compensate for this unresolved body interface.
No current result establishes standing or any rich behavior.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python audit_stance_support.py \
  --xml runs/six-leg-body/six_leg.xml --poses runs/refined-poses.json \
  --motor motor-annotations.json --output runs/support-capacity.json
.venv/bin/python audit_tibia_force.py \
  --source-xml ../FlyMimic/flymimic/assets/models/best_combined_cvt3.xml \
  --xml runs/six-leg-body/six_leg.xml --poses runs/refined-poses.json \
  --output runs/tibia-force.json
```
