# Steering-neuron responses: structured, not behavioral validation

Full MaleCNS responds differently to two identified steering-neuron types.
Direct diagnostic pulses into DNa02 predominantly affect ipsilateral leg motor
neurons; DNg13 predominantly affects contralateral ones. This distinction holds
for left/right cells, positive/negative pulses, and amplitudes 0.05, 0.2 and 0.5.
No dynamics, recruitment parameters or physical controller was changed.

The biological motivation is [Yang et al., Fine-grained descending control of
steering in walking Drosophila](https://pubmed.ncbi.nlm.nih.gov/39293446/):
unilateral DNa02 perturbation chiefly changes ipsilateral strides, whereas DNg13
perturbation chiefly changes contralateral strides. These are kinematic effects
in walking flies, **not predictions of mean motor-rate magnitude at rest**.
The [MANC circuit analysis](https://elifesciences.org/articles/96084) also
describes distinct downstream circuitry. Our neural summary is therefore an
exploratory side-bias check, not a biological pass/fail benchmark.

## Assay and results

`probe_steering_circuit.py` authenticates the official annotation file and full
released CNS model. It selects the four individually identified DNs from the
Traced/bodyId-sorted table. Each amplitude uses nine independent neural states:
one neutral control and positive/negative perturbations of each cell. Current
pulses last from 0.3 to 1.3 seconds of a two-second run. No body, sensory organ,
muscle force, learned decoder or context controller participates.

Motor differences are relative to the matched control at each timepoint. The
side-bias metric averages absolute late changes within each leg's motor cohort,
then averages the three legs per side. It is neither a fraction of spikes nor
a fraction of behavioral effect. Side labels come from the annotated soma side;
cross-specimen physiological correspondence remains a qualification.

| Stimulated cell | Ipsilateral share at amplitude 0.2 | Peak motor-rate change |
| --- | ---: | ---: |
| DNa02 L, positive | 92.11% | 0.002883 |
| DNa02 R, positive | 94.58% | 0.009044 |
| DNg13 L, positive | 12.53% | 0.001502 |
| DNg13 R, positive | 14.40% | 0.001081 |

Across all three amplitudes and both pulse signs, DNa02 ipsilateral shares
range 92.09–94.59%; DNg13 shares range 12.52–14.40%. Magnitudes grow with pulse
strength but are not left/right symmetric. These 24 pulse conditions plus three
controls are deterministic dose comparisons, not independent biological samples.
They do not show a locomotor rhythm, phase-dependent action, sensory-driven
steering, or any useful whole-body behavior.

## Rate range matters for recruitment

The pinned CNS source computes `h = min(r0, 1-r0)` and
`target = r0 + support*h*tanh(u/h)`. Support is bounded above by 1, and each rate
update is a convex interpolation toward this target. Starting from `r0`, the
rate therefore remains between `r0-h` and `r0+h`, up to numerical error.
The authenticated [operating-point receipt](evidence/cns-operating-point-001.json)
reports baseline 0.2, giving an approximate range **0–0.4**, not 0–1.

The current mean-rate muscle mapping consequently cannot reach activation 1.
This is another reason its recruitment needs physiological calibration. It does
not explain away the mechanical support tests: several poses failed even with
arbitrary inputs up to 1, and some named-pool cases failed without a ceiling.
Blindly doubling rate-to-activation gain would also change tonic co-contraction;
no such change is made here. Model rates and injected currents are not Hz or pA.

## Evidence and consequence

`evidence/trials/steering-circuit-{001,002,003}` contain the 0.2, 0.05 and 0.5
trials respectively. They ran on GPUs 0, 1 and 2; all finite runs have ended.
`verify_steering_circuit.py` reconstructs cohort identities and reported metrics,
checks zero pre-pulse lane differences, and confirms the directly stimulated
cell changes in the requested direction. It does not independently replay the
neural recurrence. Verification is recorded in
`evidence/steering-circuit-verification-001.json`.

These results argue against treating the CNS as unresponsive or discarding it
solely because the current body collapses. They do not validate its physiological
timing, phase dependence, motor magnitudes or behavioral competence. Direct DN
pulses remain a laboratory-style diagnostic and must not become an external
steering policy in the ab-initio experiment.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python probe_steering_circuit.py \
  --source ../chreatures --model ../releases/chreatures-v5-garden/model \
  --annotations data/raw/annotations.feather --motor motor-annotations.json \
  --output runs/steering-circuit --amplitude .2
python3 verify_steering_circuit.py --trials runs/steering-circuit \
  --output runs/steering-verification.json
```
