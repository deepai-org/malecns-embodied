# MaleCNS embodied fly experiment

An attempt to run a biologically grounded fly on a four-RTX-3090 machine:

**Physical environment → sensory organs → MaleCNS dynamics → motor neurons →
muscles/body → changed sensory input.**

**Status: research prototype, not a demonstrated autonomous fly.** We have run
the full curated MaleCNS graph in a physical feedback loop, but the initialized
model has not demonstrated useful walking, feeding, grooming, or flight.
No high-level behavior selector or imitation policy counts as achieving the goal.
See the current [route reassessment](REASSESSMENT.md) and
[passive-mechanics diagnostics](PASSIVE_MECHANICS.md) before extending the setup.
The identified force-update bug is now repaired and checked in a separate
[physics-step muscle host](PHYSICS_MUSCLES.md); useful behavior remains unproven.
The subsequent [sensory-transfer test](SENSORY_TRANSFER.md) finds responsive CNS
motor output but weak antagonist separation; global input amplification is not
installed as a fix.

## What currently works

- Authenticated CNS V5 execution: 165,122 neurons, 25,563,197 directed connections.
- Four physical fly bodies with retinal and body feedback, no external private
  controller/context, and motor-neuron-derived joint commands.
- Matched intact, sensory-disconnected, and motor-output-clamped trials.
- Fifteen anatomy-based left-foreleg muscle actuators tested independently.
- A six-muscle candidate interface restricted to named motor-neuron groups,
  with unit tests and open-loop replay of recorded MaleCNS motor activity.
- Named proprioceptor perturbations through the full graph: expected relative
  motor preferences, but incomplete signed responses. [Assay and caveats](PROPRIOCEPTION.md).
- A physical foreleg muscle feedback loop through the full CNS, with both
  candidate sensory polarities and matched disconnected controls.
  [Mechanism, evidence and assumptions](MUSCLE_FEEDBACK.md).
- A separate whole-body torque host and approximate named antagonist muscles
  spanning twelve joints across all six legs. Servo force removal and matched
  CNS trials are verified. [Results and unresolved mechanics](AGGREGATE_BODY.md).

## What the evidence does **not** show

The whole-body baseline uses effective joint-position servos, not identified
muscles. All conditions stay upright, including clamped motor output. Initial
motion is predominantly vertical settling; neural contributions are tiny.
The new experimental torque mode removes those servo forces, but the original
body's strong passive joint springs remain. Its muscle approximation also
mostly settles. Neither version has demonstrated useful locomotion.
The latest physics-step muscle trials remove joint springs and damping and
remain finite for four seconds, but still do not demonstrate useful behavior.

The separate muscle model has a fixed body and only one muscle-driven foreleg.
Six mappings are anatomical-name correspondences across specimens, not measured
neuromuscular weights. Equal averaging of normalized motor rates is an
uncalibrated hypothesis. Muscle replay is **open loop**, not a replacement for
the whole-body feedback experiment. Wiring alone does not determine physiology.

See [STATUS.md](STATUS.md) for the chronological audit, negative results, source
limitations, failed attempts, and next work. [evidence/trials](evidence/trials)
contains the actual trial records, including failed runs, not rendered animations.
NPZ files are numeric arrays; load them with `allow_pickle=False`.

## Reproduce

Source revisions are pinned in [sources.json](sources.json). Upstream code is
not vendored. Use a separate directory for each checkout and do not update the
pins silently. A Linux NVIDIA host is needed for the neural probes.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/emberian/chreatures.git ../chreatures
git -C ../chreatures checkout c30fbccb541ae5bc2cbdb88f7ef38848c611a8e6
git clone https://github.com/gizemozd/FlyMimic.git ../FlyMimic
git -C ../FlyMimic checkout 9ea1131626cd76f7203b74076ef8f0e9cab30bef
.venv/bin/python -m unittest -v test_neuromuscular test_muscle_feedback test_aggregate_leg_muscles
```

Download the V5 garden archive linked in `sources.json`, verify its SHA256, and
extract it into `../releases`. It contains the model tensor directory. The
scripts authenticate compressed and uncompressed tensor hashes against the
release manifest and validate the upstream array contract.

```sh
.venv/bin/python probe_cns.py --source ../chreatures \
  --model ../releases/chreatures-v5-garden/model --output neural-test.json

# Install Rust and a C compiler first. Adjust these paths to this venv.
MUJOCO_INCLUDE_DIR="$PWD/.venv/lib/python3.12/site-packages/mujoco/include" \
MUJOCO_LIB_DIR="$PWD/.venv/lib/python3.12/site-packages/mujoco" \
cargo build --locked --release --manifest-path ../chreatures/native/fly-world/Cargo.toml

.venv/bin/python run_embodied.py --source ../chreatures \
  --model ../releases/chreatures-v5-garden/model \
  --scene ../chreatures/native/fly-body/scenes/training-4/world.json \
  --output runs/intact --steps 200
```

Repeat with fresh output directories and `--condition sensory-disconnected` or
`--condition motor-disconnected`, then use `compare_trials.py` on the three
directories. `motor-disconnected` clamps neural commands to zero; it does not
remove the downstream neutral-position servo forces. All three trials use the
same seed; four residents share one world and are not independent replicates.

Other scripts expose their arguments via `--help`. GPU selection can be set via
`CUDA_VISIBLE_DEVICES`; the scripts use logical `cuda:0`. Runs are finite and
do not install cron jobs or background services. Output paths must be fresh.

`python download_data.py` downloads and verifies the three official raw data
files (~1.1 GB) under ignored `data/raw/`. The two annotation audit scripts can
regenerate the small derived tables committed here. Interrupted `.partial`
downloads are retained; inspect/move them before retrying rather than silently
overwriting them.

## Data, licensing and security

Original experiment code is AGPL-3.0-or-later. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md) for upstream attribution and separate data licenses.
The public repository intentionally excludes SSH configuration, private machine
addresses, credentials, environment secrets, virtual environments and bulk raw
connectome downloads. The original self-improvement experiment is out of scope.
