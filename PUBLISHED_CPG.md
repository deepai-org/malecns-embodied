# A connectome-derived rhythm reference

Pugliese and colleagues' [preprint, version 2](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v2)
reports motor rhythms under constant descending input, using connectome-based
networks including male CNS. This is a more directly relevant dynamics reference
than a behavioral controller. It is a preprint, and a rhythm is not a walking fly.

The [author repository](https://github.com/smpuglie/Pugliese_2026), accessed through
its older Pugliese_cpg_2025 redirect, is pinned separately at
`faee4b06869855ae0164cbf217fb6ec28ef3521b`. No existing source pin is replaced.
Its Extended Data Figure 3 notebook labels the `imac` data directory as mCNS.
The supplied front-leg network contains 4,310 neurons. It is not the full
165,122-neuron artifact used in our embodied runs.

## Independent mean-parameter probe

`probe_published_cpg.py` checks matrix row/column body IDs against the supplied
annotation table. It transposes the pre-by-post signed matrix and scales both
signs by 0.03, following the upstream reweighting function. Dynamics are:

`dR/dt = (max(200*tanh((I + W*R - 7.5)/200), 0) - R) / 0.02`

These are the default **mean** parameters, without randomized draws or cell-size
scaling. This is not a reproduction of the authors' parameter ensemble or solver.
We independently integrate with SciPy RK45 in float64. The two stimulated
conditions use a constant input of 250 to one DNg100 cell each (body IDs 10045
and 10056). A matched zero-input condition starts at the same zero state.
The recorded interval is pulse onset (0.02 s) to pulse end (1.999 s).

No oscillating input, fitted trajectory, muscle command, or physical body is
present. A descriptive motor criterion counts at least three peaks after 0.5 s
with prominence at least one model-rate unit. It is not the authors' oscillation
score and is not proof of biologically correct frequency or muscle coordination.

## Numerical failures retained

The first local attempt (`published-cpg-001`) completed the zero-input condition
but failed a nonnegative-rate check during stimulation, before saving a trace.
A second attempt limited RK45 steps to 1 ms but still failed: the minimum
sampled rate was -0.00013538, while the maximum was approximately 200. Neither
attempt is a completed trial; their empty output directories were not reused.
No negative values were silently clipped. The follow-up uses tolerances 100
times tighter on the Vast host and records the observed extrema explicitly.

## Completed Vast reference: trial 003

The CPU reference run on the existing Vast host completed in 103.13 seconds.
It used 118,920 nonzero signed edges and recorded all 130 labeled motor neurons
plus selected circuit cells. Results under the descriptive peak criterion:

| Input | Motor neurons with at least three late peaks |
| --- | ---: |
| None | 0 |
| Constant DNg100 body 10045 input | 8 |
| Constant DNg100 body 10056 input | 19 |

Inverse mean interpeak intervals span 3.40–10.05 Hz and 3.68–5.03 Hz respectively.
These are not established gait frequencies, proof of periodic limit cycles,
or validated muscle coordination. Some neurons reach the 200-unit rate cap.
Small negative numerical undershoots remain: minima -1.60e-7 and -2.46e-7.
They are recorded, not clipped. A further tolerance comparison is needed before
claiming convergence; the two earlier failed attempts are not such a comparison.

`evidence/trials/published-cpg-003/` contains the receipt and selected traces.
`verify_published_cpg.py` recomputes all motor peak counts, ranges and means,
checks the exact silent control, and authenticates the trace. Its saved result
is `evidence/published-cpg-verification-003.json`. This verifies recorded
summaries, not independent regeneration of the neural trajectories.

```sh
OPENBLAS_NUM_THREADS=1 python3 probe_published_cpg.py --source ../Pugliese_2026 \
  --output runs/published-cpg --rtol 2e-8 --atol 5e-11
python3 verify_published_cpg.py --trial runs/published-cpg \
  --output runs/published-cpg-verification.json
```

## Scope of the next decision

### Tolerance refinement completed

Trial 004 reran the same source, mean parameters, input and 1-ms maximum step
on Vast with `rtol=2e-9, atol=5e-12` (ten times tighter than trial 003).
The silent control remained exactly zero. Repeated-peak motor counts remained
8 and 19. Comparing all recorded motor samples, maximum differences were
0.01835 and 0.36310 model-rate units; RMS differences were 0.000470 and 0.002914.
The larger maximum is about 0.182% of the 200-unit rate cap, not necessarily
0.182% of each neuron's response. Negative undershoots decreased to -2.36e-8
and -3.26e-9. There was no clipping or substitution of failed results.

This supports the repeated-peak observation under tolerance refinement; it is
not exact trajectory agreement, an asymptotic convergence proof, or biological
validation. Both complete traces are retained. `compare_cpg_tolerances.py`
authenticates matching source/input identities and computes samplewise errors.
Results are in `evidence/cpg-tolerance-comparison-001.json`; trace verification
for the refined run is `evidence/published-cpg-verification-004.json`.

The next comparison should separate two source-confirmed changes: direct signed
synaptic-count scaling versus per-target input normalization, and rectified
absolute rates versus baseline-centered dynamics. Do not interpret the reference
as evidence that one arbitrary global-gain increase will repair the current fly.

If this reference produces numerically stable rhythms, compare its signed
synaptic-count scaling and rectified dynamics with our current normalized,
baseline-centered dynamics. Preserve cell identities and connectivity when
testing an alternative on the full MaleCNS graph. Do not install the extracted
CPG as an external oscillator or claim constant DN stimulation is autonomous
sensory-driven behavior. Mechanics and sensory transduction remain unresolved.
