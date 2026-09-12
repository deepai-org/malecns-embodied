"""Verify disturbance controls and report recovery without a behavioral verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from neuromuscular import NamedMuscleDrive
from run_muscle_feedback import position_current


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trial','mapping','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--xml',type=Path,help='Also replay physical intervals with MuJoCo')
    a = p.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    r = json.loads((a.trial/'result.json').read_text())
    assert r['completed'] and r['steps'] == 200 and r['pulse_ticks'] == [80,85]
    assert sha(a.mapping) == r['input_sha256']['mapping']
    assert sha(a.trial/'trace.npz') == r['trace_sha256']
    with np.load(a.trial/'trace.npz',allow_pickle=False) as z:
        t = {k:z[k] for k in z.files}
    assert all(np.isfinite(v).all() for v in t.values())
    assert t['qpos'].shape[:2] == (201,12)
    np.testing.assert_array_equal(t['time'],np.arange(201)*.01)
    drive = NamedMuscleDrive(json.loads(a.mapping.read_text()),t['motor_rows'])
    commands = np.stack([[drive(t['motor_rates'][tick,:,lane]) for lane in range(12)]
                         for tick in range(200)])
    np.testing.assert_allclose(commands,t['controls'],rtol=0,atol=6e-8)
    qi = r['qpos_index']
    comparisons = []
    for base in range(0,12,3):
        sham, feedback, yoked = base,base+1,base+2
        conds = r['conditions'][base:base+3]
        assert [c['mode'] for c in conds] == ['sham','feedback','yoked-sensory']
        assert len({(c['polarity'],c['sign']) for c in conds}) == 1
        for lane in (feedback,yoked):
            np.testing.assert_array_equal(t['qpos'][:81,lane],t['qpos'][:81,sham])
        np.testing.assert_array_equal(t['sensory_current'][:,yoked],t['sensory_current'][:,sham])
        np.testing.assert_array_equal(t['motor_rates'][:,:,yoked],t['motor_rates'][:,:,sham])
        np.testing.assert_array_equal(t['controls'][:,yoked],t['controls'][:,sham])
        for lane in (sham,feedback,yoked):
            expected_torque = np.zeros(200)
            if lane != sham:
                expected_torque[80:85] = conds[0]['sign']*r['torque_model_units']
            np.testing.assert_array_equal(t['external_torque'][:,lane],expected_torque)
            sensed = sham if lane == yoked else lane
            expected_input = np.stack([position_current(q,r['joint_range'],
                r['amplitude_model_units'],conds[0]['polarity']) for q in t['qpos'][:-1,sensed,qi]])
            np.testing.assert_array_equal(expected_input,t['sensory_current'][:,lane])
        f = t['qpos'][:,feedback,qi]-t['qpos'][:,sham,qi]
        y = t['qpos'][:,yoked,qi]-t['qpos'][:,sham,qi]
        area_f,area_y = (float(np.sum(np.abs(v[85:]))*.01) for v in (f,y))
        comparisons.append(dict(polarity=conds[0]['polarity'],sign=conds[0]['sign'],
            peak_tibia_displacement_feedback_rad=float(np.max(np.abs(f))),
            peak_tibia_displacement_yoked_rad=float(np.max(np.abs(y))),
            postpulse_abs_error_feedback_rad_s=area_f,
            postpulse_abs_error_yoked_rad_s=area_y,
            relative_postpulse_error_reduction=(area_y-area_f)/area_y if area_y else None,
            final_displacement_feedback_rad=float(f[-1]),
            final_displacement_yoked_rad=float(y[-1]),
            feedback_vs_yoked_tibia_max_rad=float(np.max(np.abs(f-y))),
            feedback_vs_yoked_control_max=float(np.max(np.abs(t['controls'][:,feedback]-t['controls'][:,yoked])))))
    replay = None
    if a.xml:
        import mujoco
        assert sha(a.xml) == r['input_sha256']['xml']
        assert mujoco.__version__ == r['mujoco']
        m = mujoco.MjModel.from_xml_path(str(a.xml))
        b = mujoco.MjData(m)
        spec = mujoco.mjtState(r['integration_state_spec'])
        errors = []
        for tick in (79,80,84,85,120,199):
            for lane in range(12):
                mujoco.mj_setState(m,b,t['integration'][tick,lane],spec)
                b.ctrl[:] = t['controls'][tick,lane]
                b.qfrc_applied[:] = 0
                b.qfrc_applied[r['dof_index']] = t['external_torque'][tick,lane]
                for _ in range(round(.01/m.opt.timestep)):
                    mujoco.mj_step(m,b)
                errors.append(float(np.max(np.abs(b.qpos-t['qpos'][tick+1,lane]))))
        assert max(errors) < 1e-10
        replay = dict(intervals=len(errors),qpos_max_error=max(errors))
    out = dict(verified=True,verifier_sha256=sha(Path(__file__)),
        trial_receipt_sha256=sha(a.trial/'result.json'), comparisons=comparisons,
        control_reconstruction_max_error=float(np.max(np.abs(commands-t['controls']))),
        physical_replay=replay,
        caveat='Recovery relative to moving sham, not a biological target; no calibrated reflex or whole-fly success claim.')
    with a.output.open('x') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
