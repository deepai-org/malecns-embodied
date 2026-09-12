"""Feed measured simulation motor rates into named muscles: component replay only.

This is intentionally OPEN LOOP: these muscle movements do not feed the source
neural trial. It tests the candidate efferent interface, not whole-fly behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from model_loader import load_arrays
from neuromuscular import NamedMuscleDrive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--mapping', type=Path, required=True)
parser.add_argument('--xml', type=Path, required=True)
parser.add_argument('--trial', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(args.source))
arrays, _, model_hash = load_arrays(args.model)
mapping = json.loads(args.mapping.read_text())
if hashlib.sha256(args.xml.read_bytes()).hexdigest() != mapping['xml_sha256']:
    raise ValueError('muscle geometry differs from mapping audit')
source = json.loads((args.trial / 'result.json').read_text())
if not source['completed'] or source['model_manifest_sha256'] != model_hash:
    raise ValueError('neural trace model identity mismatch')
with np.load(args.trial / 'trace.npz', allow_pickle=False) as archive:
    rates = archive['motor_rates'][:, :, 0].copy()
    dt = np.diff(archive['time'])
if not np.allclose(dt, .01, rtol=0, atol=1e-10):
    raise ValueError('unexpected neural timestep')
drive = NamedMuscleDrive(mapping, arrays['atlas.motor_rows'])
model = mujoco.MjModel.from_xml_path(str(args.xml))
for i, name in enumerate(drive.names):
    if model.actuator(i).name != name:
        raise ValueError('muscle order differs')
data = mujoco.MjData(model)
outputs = {}
for condition in ('intact-drive', 'zero-neural-drive'):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    qpos, controls = [data.qpos.copy()], []
    for row in rates:
        activation = drive(row) if condition == 'intact-drive' else np.zeros(model.nu)
        data.ctrl[:] = activation
        for _ in range(100):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError('nonfinite muscle state')
        qpos.append(data.qpos.copy())
        controls.append(activation.copy())
    outputs[condition+'_qpos'] = np.asarray(qpos)
    outputs[condition+'_controls'] = np.asarray(controls)
np.savez_compressed(args.output / 'trace.npz', **outputs)
report = dict(kind='open-loop replay of one resident motor activity into six named muscle candidates',
    source_trace_sha256=hashlib.sha256((args.trial/'trace.npz').read_bytes()).hexdigest(),
    mapping_sha256=hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
    completed=True, steps=len(rates), simulated_seconds=len(rates)*.01,
    maximum_joint_difference_radians=float(np.max(np.abs(outputs['intact-drive_qpos']-outputs['zero-neural-drive_qpos']))),
    assumption='equal mean normalized motor rate drives muscle activation; uncalibrated',
    unresolved_muscles='zero neural drive; upstream minimum activation and passive mechanics remain',
    behavior_claim='none; no feedback from this muscle component to the source neural trial')
(args.output/'result.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
