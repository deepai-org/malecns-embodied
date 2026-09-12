# Physical screen → retinal samples → CNS: weak downstream response

The direct steering-neuron pulse assay bypassed vision. This test instead sends
screen images through the original native physical world's eye rays and then
through the existing retinal projection and full MaleCNS recurrence. It injects
no current into descending neurons and supplies no motor command.

**Result:** visual changes reach the retinal samples, but descending/motor
changes are very small. This is not evidence of visual steering or useful
behavior. It also does not establish why the response is weak.

## Fixed-body visual assay

`probe_screen_vision.py` creates four separate, identically seeded physical
worlds. Each contains the original four residents and physical screen. Physics
is never advanced: body poses and world time remain fixed, while the neural
clock advances for two seconds. This deliberately isolates a sensory pathway;
it is not a closed physical feedback loop or a standing test.

The four conditions are gray, stationary sinusoidal grating, positive-phase
motion and negative-phase motion. Gratings are displayed from neural time 0.3
to 1.3 s at 128×64 pixels, four cycles across the screen, luminance 0.5±0.45 and
two temporal cycles/s. Phase signs are screen-image coordinates, not asserted
body-relative turn directions. The native screen and scene determine visibility,
occlusion and retinal samples. No image is painted directly onto neural rows.
BODY inputs stay at their model means and external context is zero.

| Contrast against gray | Peak retinal change | Peak motor-rate change | Largest steering-cell change |
| --- | ---: | ---: | ---: |
| Stationary grating | 0.44946 | 7.15e-7 | 1.19e-7 |
| Positive-phase motion | 0.44983 | 3.58e-7 | 5.96e-8 |
| Negative-phase motion | 0.44983 | 3.87e-7 | 4.47e-8 |

Only 123 of 1,771 distinct site indices change, taking the union across all four
residents. Thus this is a limited-field screen stimulus, not panoramic optic
flow. Moving-versus-stationary motor differences peak at about 1.09e-6, and
steering-cell differences at about 1.49e-7. Some descending differences are only
one or a few float32 increments near the 0.2 baseline; they must not be treated
as established direction selectivity or a biologically meaningful signal.

All 4,107 receptor rows have active fast-channel labels in the authenticated
artifact. Source graph construction includes histamine among its negative-sign
fast transmitters. This rules out the specific suggestion that the channel
builder simply omits histamine; it does not validate retinal wiring, dynamics,
synaptic gains or visual physiology.

## Evidence and consequence

The receipt and numeric trace are in `evidence/trials/screen-vision-001/`.
`verify_screen_vision.py` checks finite samples, dimensions, identities, input
bounds and matching before the stimulus, and reconstructs the reported retinal,
motor and DN contrasts. Pre-stimulus contrasts are exactly zero.
`evidence/screen-vision-verification-001.json` records those checks. It is not
an independent replay of the physical ray sampler or neural recurrence.

The finite experiment completed and its four temporary world processes were
closed. Existing bodies, source assets and all CNS parameters remain unchanged.
The six-leg muscle runner has not gained a vision adapter from this assay.

The direct-DN result must not be extrapolated to visually driven control. Before
adding vision-dependent behavior claims, compare broader visual-field stimulation
and intermediate visual-neuron responses to distinguish limited input coverage
from attenuation in the transduction/CNS pathway. Do not compensate by making
an external policy stimulate steering DNs.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python probe_screen_vision.py \
  --source ../chreatures --model ../releases/chreatures-v5-garden/model \
  --scene ../chreatures/native/fly-body/scenes/training-4/world.json \
  --steering-receipt evidence/trials/steering-circuit-001/result.json \
  --output runs/screen-vision
python3 verify_screen_vision.py --trial runs/screen-vision \
  --output runs/screen-vision-verification.json
```
