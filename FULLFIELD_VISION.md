# Full-field vision: coverage is not the only limitation

The physical-screen assay stimulated only a small part of the sampled retina.
`probe_fullfield_vision.py` removes that coverage limitation diagnostically:
every retinal site receives a prescribed full-field signal through the existing
receptor projection. It **bypasses physical ray sampling**, runs no body and
does not provide descending-neuron current or motor control.

Five independent full-CNS states receive gray, bright (0.95), dark (0.05), and
two moving sinusoidal gratings during seconds 0.3–1.3 of a two-second run.
Gratings use the existing retinal camera-local azimuths independently per eye;
they are not claimed to reproduce a calibrated world-relative optomotor stimulus.
BODY stays at the model mean and external context is zero. No parameter changes.

## Results

Bright-field peak absolute changes relative to matched gray control:

| Readout group | Peak normalized rate change |
| --- | ---: |
| Receptors | 0.1833 |
| L1 | 0.0941 |
| L2 | 0.0718 |
| L3 | 0.0663 |
| Mi1 | 0.0179 |
| Tm3 | 0.0202 |
| T4-prefixed types | 0.00719 |
| T5-prefixed types | 0.00820 |
| DNa02 | 0.00000598 |
| DNg13 | 0.00000508 |
| All motor neurons | 0.00001125 |

Dark-field results are similar in absolute magnitude. Moving gratings produce
receptor peaks 0.148–0.159, T4/T5 peaks about 0.0037–0.0041, steering-cell peaks
below 6.2e-7 and motor peaks below 2.7e-6. Some late motor averages are near
float32 resolution. No claim of direction selectivity is made.

The visible responses in receptor and named visual groups show that the whole
visual input is not simply absent. Expanding coverage increases downstream
responses compared with the screen assay, but does not produce substantial
steering/motor activity in these resting-state conditions. Limited screen
coverage therefore does not fully explain the earlier weak response.

These groups are **not asserted to be a single serial pathway**. Different group
sizes, parallel routes, stimulus tuning, signs, synaptic gain and network state
all affect the maxima. The table cannot localize a faulty synapse, establish a
biological gain ratio, or justify multiplying a neural gain. No walking-state
or physiological recording comparison has been performed.

## Evidence and next decision

The subsequent `analyze_motion_selectivity.py` analysis separates T4/T5 subtypes
and sides and compares AC RMS over the last complete stimulus cycle. Saved
results are in `evidence/motion-selectivity-001.json`. Median absolute
opposite-phase asymmetries across groups are about 0.026–0.102; these are not
physiological direction-selectivity indices. Stimuli remain unregistered to
preferred world directions, so this test cannot establish absence of motion
tuning. It changes no dynamics or stimulus parameters.

`evidence/trials/fullfield-vision-001/` records the full stimuli, selected-neuron
time series, exact body IDs/row indices and source hashes. The trace is about
60 MB compressed; it is experiment evidence, not a bulk source dataset.
`verify_fullfield_vision.py` reconstructs all 44 condition/group summaries and
checks exactly matched pre-stimulus states. This is a recording/analysis check,
not independent CNS regeneration. Verification is saved in
`evidence/fullfield-vision-verification-001.json`.

The result argues for examining visual-to-descending dynamics and stimulus/state
dependence, not treating the successful direct-DN pulse assay as an operational
visual system. It adds no external controller and does not resolve muscle
support, sensory-driven movement, or rich behavior. The finite assay has ended.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python probe_fullfield_vision.py \
  --source ../chreatures --model ../releases/chreatures-v5-garden/model \
  --annotations data/raw/annotations.feather \
  --scene ../chreatures/native/fly-body/scenes/training-4/world.json \
  --output runs/fullfield-vision
python3 verify_fullfield_vision.py --trial runs/fullfield-vision \
  --output runs/fullfield-verification.json
```
