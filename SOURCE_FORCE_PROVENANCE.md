# Source force parameters: no established common conversion fix

The [FlyMimic paper, methods and supplement](https://arxiv.org/html/2509.06426v2)
initializes maximum force from cross-sectional area and specific tension
28 mN/mm², then optimizes a factor in the range 0.3–3. It models the fast tibia
flexor and extensor; therefore the low tip-force ratio cannot simply be dismissed
as intentionally modeling only a weak, slow flexor. Measured physiological
force calibration remains distinct from fitting limb kinematics.

The [authors' project description](https://gizemozd.github.io/fly_mimic/)
states that parameter fitting and imitation training excluded contact forces.
Its ground demonstration uses muscle-actuated front legs and torque-actuated
middle/hind legs for support. That demonstration is not evidence that six copies
of the front-leg assembly have sufficient whole-body support capacity.

`audit_source_force_parameters.py` compares the 15 matched muscle names in the
pinned `best_combined_full.osim` and `best_combined_cvt3.xml`. MuJoCo/OpenSim
Fmax ratios vary from 0.210811 to 1.972411; the tibia flexor ratio is about 1.64.
Both files use gravity magnitudes near 9800 in their numerical coordinates.
OpenSim nevertheless declares length/force units `m`/`N`, so its unit labels
alone do not establish an SI interpretation for this scaled model.

These are two published artifacts, not an independently reproduced conversion
pipeline. Differences may include parameter fitting, and no exact lineage is
established. The comparison does **not** prove conversion correctness, but it
provides no evidence for applying a common missing force multiplier. All source
and derived assets remain unchanged. The linked original optimization repository
still returned HTTP 404 during this check.

Receipt: `evidence/source-force-parameters-001.json` records both source hashes,
the executed script hash, force/fiber/tendon parameters, and MuJoCo activation
time constants. No anatomy, force scale, recruitment map or CNS parameter was
changed based on this audit.

```sh
python3 audit_source_force_parameters.py \
  --opensim ../FlyMimic/flymimic/assets/models/opensim/best_combined_full.osim \
  --mujoco ../FlyMimic/flymimic/assets/models/best_combined_cvt3.xml \
  --output runs/source-force-parameters.json
```
