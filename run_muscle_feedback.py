"""Exploratory physical foreleg ↔ full MaleCNS loop, not whole-fly behavior.

Two uncalibrated opponent position encodings are tested, not selected for a
desired movement. Six named muscles receive only anatomically matched MN rates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from model_loader import load_arrays
from neuromuscular import NamedMuscleDrive


def position_current(angle, limits, amplitude, polarity):
    """Candidate population transduction, not measured single-cell tuning."""
    if not np.isfinite(angle) or not 0 < amplitude <= 1 or polarity not in (-1, 1):
        raise ValueError('invalid sensory parameters')
    lo, hi = limits
    if not np.isfinite(limits).all() or hi <= lo:
        raise ValueError('invalid anatomical range')
    u = float(np.clip((angle-lo)/(hi-lo), 0, 1))
    return amplitude*np.array([u, 1-u] if polarity == 1 else [1-u, u])


def main():
    import mujoco
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'mapping', 'sensory', 'xml', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--amplitude', type=float, default=.2)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.amplitude <= 1:
        raise ValueError('invalid experiment length or amplitude')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays, _, model_hash = load_arrays(args.model)
    mapping = json.loads(args.mapping.read_text())
    sensory = json.loads(args.sensory.read_text())
    if mapping['xml_sha256'] != hashlib.sha256(args.xml.read_bytes()).hexdigest():
        raise ValueError('muscle geometry differs from audited mapping')
    if mapping['neuron_source_sha256'] != sensory['source_sha256']:
        raise ValueError('annotation sources differ')
    drive = NamedMuscleDrive(mapping, arrays['atlas.motor_rows'])
    physics = mujoco.MjModel.from_xml_path(str(args.xml))
    if physics.nu != len(drive.names) or any(physics.actuator(i).name != name
                                            for i, name in enumerate(drive.names)):
        raise ValueError('muscle inventory mismatch')
    if np.any(physics.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE):
        raise ValueError('non-muscle actuator')
    substeps = round(.01/physics.opt.timestep)
    if not np.isclose(substeps*physics.opt.timestep, .01, rtol=0, atol=1e-12):
        raise ValueError('physical and neural timesteps incompatible')
    joint = physics.joint('joint_LFTibia_pitch')
    qindex = int(joint.qposadr[0])
    limits = joint.range.copy()
    groups = [[r for r in sensory['neurons'] if r['type'] == kind and
               r['entryNerve'] == 'ProLN' and (r['somaSide'] or r['rootSide']) == 'L']
              for kind in ('SNpp50', 'SNpp51')]
    for group in groups:
        if not group or any(r['cns_row'] not in arrays['atlas.body_rows'] or
                            r['cns_row'] in arrays['atlas.motor_rows'] for r in group):
            raise ValueError('invalid afferent boundary')
    conditions = [(mode, polarity) for polarity in (1, -1)
                  for mode in ('feedback', 'frozen-sensory', 'zero-motor')]
    bodies = [mujoco.MjData(physics) for _ in conditions]
    for body in bodies:
        mujoco.mj_resetDataKeyframe(physics, body, 0)
    initial_angle = float(bodies[0].qpos[qindex])
    torch.set_num_threads(8)
    device = torch.device('cuda:0')
    cns = AnatomicalCNS(arrays, device=device).eval()
    motor_rows = torch.as_tensor(np.asarray(arrays['atlas.motor_rows'], dtype=np.int64), device=device)
    state = cns.initial_state(len(conditions))
    trace = dict(qpos=[np.stack([b.qpos.copy() for b in bodies])],
                 motor_rates=[], controls=[], sensory_current=[])
    start = time.monotonic()
    with torch.inference_mode():
        neutral = cns.neutral_current(len(conditions)).contiguous()
        for tick in range(args.steps):
            current = neutral.clone()
            inputs = []
            for lane, ((mode, polarity), body) in enumerate(zip(conditions, bodies)):
                angle = initial_angle if mode == 'frozen-sensory' else float(body.qpos[qindex])
                values = position_current(angle, limits, args.amplitude, polarity)
                inputs.append(values)
                for group, value in zip(groups, values):
                    current[[r['cns_row'] for r in group], lane] += float(value)
            state = cns.step_from_current(current.contiguous(), state, neutral=neutral)
            if not all(torch.isfinite(v).all().item() for v in state.fields()):
                raise RuntimeError(f'nonfinite neural state at {tick}')
            rates = state.rates[motor_rows].cpu().numpy().copy()
            controls = []
            for lane, ((mode, _), body) in enumerate(zip(conditions, bodies)):
                body.ctrl[:] = 0 if mode == 'zero-motor' else drive(rates[:, lane])
                controls.append(body.ctrl.copy())
                before = body.time
                for _ in range(substeps):
                    mujoco.mj_step(physics, body)
                if (not np.isfinite(body.qpos).all() or not np.isfinite(body.qvel).all()
                        or not np.isclose(body.time-before, .01, rtol=0, atol=1e-10)):
                    raise RuntimeError(f'nonfinite or reset physics at {tick}, lane {lane}')
            trace['qpos'].append(np.stack([b.qpos.copy() for b in bodies]))
            trace['motor_rates'].append(rates)
            trace['controls'].append(controls)
            trace['sensory_current'].append(inputs)
    trace = {k: np.asarray(v) for k,v in trace.items()}
    trace['time'] = np.arange(args.steps+1)*.01
    np.savez_compressed(args.output/'trace.npz', **trace)
    comparisons = {}
    for first in (0, 3):
        comparisons[str(conditions[first][1])] = dict(
            feedback_vs_frozen_qpos_max_rad=float(np.max(np.abs(trace['qpos'][:,first]-trace['qpos'][:,first+1]))),
            feedback_vs_zero_motor_qpos_max_rad=float(np.max(np.abs(trace['qpos'][:,first]-trace['qpos'][:,first+2]))),
            feedback_vs_frozen_motor_rate_max=float(np.max(np.abs(trace['motor_rates'][:,:,first]-trace['motor_rates'][:,:,first+1]))),
            sensory_current_change_max=float(np.max(np.abs(trace['sensory_current'][:,first]-trace['sensory_current'][0,first]))))
    receipt = dict(completed=True, kind='exploratory fixed-base muscle feedback loop',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_manifest_sha256=model_hash,
        input_sha256={name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest()
                      for name in ('mapping', 'sensory', 'xml')},
        conditions=conditions, steps=args.steps, simulated_seconds=args.steps*.01,
        wall_seconds=time.monotonic()-start, amplitude_model_units=args.amplitude,
        joint='joint_LFTibia_pitch', joint_range_radians=limits.tolist(),
        afferents=groups, comparisons=comparisons,
        assumptions=['SNpp50/51 opponent clipped-linear position currents; both polarity hypotheses retained.',
                     'All cells within each subtype share tuning; no strain mechanics or physiological calibration.',
                     'Equal mean normalized MN rates drive six named muscle cohorts; nine muscles unresolved.',
                     'Other CNS inputs stay at model neutral values; no learned body decoder or external context.',
                     'No external perturbation: initial-state motion arises from gravity, muscle and passive mechanics.',
                     'Fixed-base foreleg only. No walking, whole-fly behavior or biological validation claimed.'])
    (args.output/'result.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
