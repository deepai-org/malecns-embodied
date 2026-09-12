"""Check recorded actuator boundaries and compare matched whole-body trials."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from aggregate_leg_muscles import AggregateLegMuscles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--trials',type=Path,default=Path('evidence/trials'))
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    root = Path(__file__).parent
    scene = json.loads(a.scene.read_text())
    annotation = json.loads((root/'motor-annotations.json').read_text())
    reference = json.loads((a.trials/'proprioceptor-circuit-004/result.json').read_text())
    with np.load(a.trials/'proprioceptor-circuit-004/trace.npz',allow_pickle=False) as archive:
        motor_rows = archive['motor_rows'].copy()
    names = ('aggregate-body-001','aggregate-body-zero-001','aggregate-body-frozen-001','aggregate-body-reverse-001')
    data, reports = {}, {}
    for name in names:
        receipt = json.loads((a.trials/name/'result.json').read_text())
        assert receipt['completed'] and receipt['model_manifest_sha256']==reference['model_manifest_sha256']
        assert receipt['scene_sha256']==hashlib.sha256(a.scene.read_bytes()).hexdigest()
        for key,file in (('runner_sha256','run_aggregate_body.py'),('muscle_code_sha256','aggregate_leg_muscles.py'),
                         ('motor_annotation_sha256','motor-annotations.json')):
            assert receipt[key]==hashlib.sha256((root/file).read_bytes()).hexdigest()
        with np.load(a.trials/name/'trace.npz',allow_pickle=False) as archive:
            trace = {k:archive[k].copy() for k in archive.files}
        assert all(np.isfinite(v).all() for v in trace.values())
        assert len(trace['time'])==receipt['steps']+1
        np.testing.assert_allclose(np.diff(trace['time']),.01,rtol=0,atol=1e-10)
        drive = AggregateLegMuscles(annotation,motor_rows,scene,receipt['polarity'])
        max_error = 0.
        for tick,rates in enumerate(trace['motor_rates']):
            expected = drive.step(rates,trace['qpos'][tick],trace['qvel'][tick],
                                  zero_motor=receipt['condition']=='zero-motor')
            difference = float(np.max(np.abs(expected-trace['command'][tick])))
            max_error = max(max_error,difference)
            np.testing.assert_allclose(expected,trace['command'][tick],rtol=0,atol=1e-7)
            np.testing.assert_allclose(drive.activation,trace['activation'][tick],rtol=0,atol=1e-7)
        channels = {g['channel'] for g in drive.groups}
        unsupported_channels = [i for i in range(92) if i not in channels]
        np.testing.assert_array_equal(trace['command'][:,:,unsupported_channels],0)
        unsupported_actuators = [body['actuators'][i] for body in scene['bodies'] for i in range(90) if i not in channels]
        np.testing.assert_array_equal(trace['actuator_force'][1:,unsupported_actuators],0)
        if receipt['condition']=='zero-motor':
            np.testing.assert_array_equal(trace['command'],0)
            np.testing.assert_array_equal(trace['actuator_force'][1:],0)
        movement = np.linalg.norm(np.diff(trace['thorax_position'][100:,:,:2],axis=0),axis=2).sum(axis=0)
        reports[name] = dict(maximum_command_recomputation_error=max_error,
                            horizontal_path_after_one_second_mm=movement.tolist(),
                            minimum_thorax_up=float(trace['thorax_up'].min()),
                            wall_seconds=receipt['wall_seconds'])
        data[name] = trace
    intact = data[names[0]]
    for name in names[1:]:
        np.testing.assert_array_equal(intact['qpos'][0],data[name]['qpos'][0])
        np.testing.assert_allclose(intact['time'],data[name]['time'],rtol=0,atol=1e-10)
        reports[name]['maximum_thorax_difference_from_intact_mm'] = float(np.linalg.norm(
            intact['thorax_position']-data[name]['thorax_position'],axis=2).max())
    xml = ET.parse(a.scene.parent/scene['scene_xml'])
    joints = [j for j in xml.findall('.//worldbody//joint') if j.get('type')!='free']
    passive = dict(joints=len(joints),stiffness_values=sorted({float(j.get('stiffness','0')) for j in joints}),
                   damping_values=sorted({float(j.get('damping','0')) for j in joints}))
    report = dict(checked=True,trials=reports,passive_joint_audit=passive,
                  note='Force boundaries and recorded transformations checked; not evidence of useful behavior.',
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
