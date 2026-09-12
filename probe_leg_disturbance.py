"""Physical tibia disturbance with sham-yoked sensory control; not fly behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from model_loader import load_arrays
from neuromuscular import NamedMuscleDrive
from run_muscle_feedback import position_current


def main():
    import mujoco
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'mapping', 'sensory', 'xml', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--torque', type=float, default=.1)
    a = p.parse_args()
    if not np.isfinite(a.torque) or a.torque <= 0:
        raise ValueError('torque must be finite and positive')
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays, _, model_hash = load_arrays(a.model)
    mapping = json.loads(a.mapping.read_text())
    sensory = json.loads(a.sensory.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert mapping['xml_sha256'] == sha(a.xml)
    assert mapping['neuron_source_sha256'] == sensory['source_sha256']
    drive = NamedMuscleDrive(mapping, arrays['atlas.motor_rows'])
    m = mujoco.MjModel.from_xml_path(str(a.xml))
    assert m.nu == 15 and all(m.actuator(i).name == n for i,n in enumerate(drive.names))
    assert np.all(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    substeps = round(.01/m.opt.timestep)
    assert np.isclose(substeps*m.opt.timestep, .01, rtol=0, atol=1e-12)
    j = m.joint('joint_LFTibia_pitch')
    qi, vi = int(j.qposadr[0]), int(j.dofadr[0])
    groups = [[r['cns_row'] for r in sensory['neurons'] if r['type'] == typ
               and r['entryNerve'] == 'ProLN' and (r['somaSide'] or r['rootSide']) == 'L']
              for typ in ('SNpp50', 'SNpp51')]
    assert all(groups)
    assert all(r in arrays['atlas.body_rows'] and r not in arrays['atlas.motor_rows']
               for g in groups for r in g)
    conditions = [dict(polarity=pol, sign=sign, mode=mode)
                  for pol in (1,-1) for sign in (1,-1)
                  for mode in ('sham', 'feedback', 'yoked-sensory')]
    bodies = [mujoco.MjData(m) for _ in conditions]
    for b in bodies:
        mujoco.mj_resetDataKeyframe(m, b, 0)
    torch.set_num_threads(8)
    cns = AnatomicalCNS(arrays, device=torch.device('cuda:0')).eval()
    rows = torch.as_tensor(np.asarray(arrays['atlas.motor_rows'], dtype=np.int64), device='cuda:0')
    state = cns.initial_state(len(bodies))
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    def snapshot(b):
        v = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, b, v, spec)
        return v
    trace = dict(qpos=[np.stack([b.qpos.copy() for b in bodies])],
                 integration=[np.stack([snapshot(b) for b in bodies])],
                 controls=[], motor_rates=[], sensory_current=[], external_torque=[])
    start = time.monotonic()
    with torch.inference_mode():
        neutral = cns.neutral_current(len(bodies)).contiguous()
        for tick in range(200):
            current = neutral.clone()
            inputs = []
            for lane, (cond,b) in enumerate(zip(conditions,bodies)):
                sensed = bodies[lane-lane%3] if cond['mode'] == 'yoked-sensory' else b
                vals = position_current(float(sensed.qpos[qi]), j.range, .2, cond['polarity'])
                inputs.append(vals)
                for g,v in zip(groups,vals):
                    current[g,lane] += float(v)
            state = cns.step_from_current(current.contiguous(), state, neutral=neutral)
            assert all(torch.isfinite(v).all().item() for v in state.fields())
            rates = state.rates[rows].cpu().numpy().copy()
            controls, torques = [], []
            for lane,(cond,b) in enumerate(zip(conditions,bodies)):
                b.ctrl[:] = drive(rates[:,lane])
                torque = cond['sign']*a.torque if 80 <= tick < 85 and cond['mode'] != 'sham' else 0.
                b.qfrc_applied[:] = 0
                b.qfrc_applied[vi] = torque
                controls.append(b.ctrl.copy())
                torques.append(torque)
                for _ in range(substeps):
                    mujoco.mj_step(m,b)
                assert np.isfinite(b.qpos).all() and np.isfinite(b.qvel).all()
                assert abs(b.time-(tick+1)*.01) < 1e-10
            trace['qpos'].append(np.stack([b.qpos.copy() for b in bodies]))
            trace['integration'].append(np.stack([snapshot(b) for b in bodies]))
            for key,value in (('controls',controls),('motor_rates',rates),
                              ('sensory_current',inputs),('external_torque',torques)):
                trace[key].append(value)
    trace = {k:np.asarray(v) for k,v in trace.items()}
    trace['time'] = np.arange(201)*.01
    trace['motor_rows'] = np.asarray(arrays['atlas.motor_rows'])
    np.savez_compressed(a.output/'trace.npz', **trace)
    receipt = dict(completed=True, conditions=conditions, steps=200,
        pulse_ticks=[80,85], torque_model_units=a.torque, amplitude_model_units=.2,
        qpos_index=qi, dof_index=vi, joint_range=j.range.tolist(), afferent_rows=groups,
        model_manifest_sha256=model_hash, runner_sha256=sha(Path(__file__)),
        input_sha256={n:sha(getattr(a,n)) for n in ('mapping','sensory','xml')},
        dependency_sha256={n:sha(Path(__file__).with_name(n)) for n in
            ('model_loader.py','neuromuscular.py','run_muscle_feedback.py')},
        trace_sha256=sha(a.output/'trace.npz'), mujoco=mujoco.__version__,
        integration_state_spec=int(spec), wall_seconds=time.monotonic()-start,
        assumptions=['Uncalibrated torque in model units, applied physically for 50ms; not a controller.',
            'Yoked CNS senses its matched sham body, not its own perturbed body.',
            'All 15 foreleg muscles use supplied named pools and raw mean rates.',
            'Both uncalibrated position-current polarities retained; organ strain is not modeled.',
            'Fixed-base foreleg; no parameter fitting or whole-fly behavioral claim.'])
    (a.output/'result.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(completed=True,wall_seconds=receipt['wall_seconds'])))


if __name__ == '__main__':
    main()
