"""Reconstruct named transduction and muscle recruitment from embodied traces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from named_leg_senses import NamedLegPosition
from aggregate_leg_muscles import AggregateLegMuscles
from physics_muscles import muscle_coefficients


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=Path(__file__).parent
    scene=json.loads(a.source_scene.read_text())
    variant=json.loads((root/'evidence/passive-variant-none-001.json').read_text())
    assert sha(a.source_scene)==variant['source_scene_sha256']
    sensory=json.loads((root/'proprioceptor-annotations.json').read_text())
    motor=json.loads((root/'motor-annotations.json').read_text())
    ref=json.loads((root/'evidence/trials/proprioceptor-circuit-004/result.json').read_text())
    with np.load(root/'evidence/trials/proprioceptor-circuit-004/trace.npz',allow_pickle=False) as t:
        motor_rows=t['motor_rows'].copy()
    traces={}; reports={}
    for mode in ('plus','minus','frozen','zero'):
        directory=root/f'evidence/trials/named-senses-{mode}-001'
        r=json.loads((directory/'result.json').read_text())
        assert r['completed'] and r['model_manifest_sha256']==ref['model_manifest_sha256']
        assert r['scene_sha256']==variant['derived_scene_sha256']
        for key,file in [('runner_sha256','run_named_senses.py'),('sensory_adapter_sha256','named_leg_senses.py'),
                         ('sensory_annotation_sha256','proprioceptor-annotations.json'),('muscle_code_sha256','aggregate_leg_muscles.py'),
                         ('force_adapter_sha256','physics_muscles.py'),('motor_annotation_sha256','motor-annotations.json')]:
            assert r[key]==sha(root/file)
        with np.load(directory/'trace.npz',allow_pickle=False) as archive:
            t={k:archive[k].copy() for k in archive.files}
        assert all(np.isfinite(v).all() for v in t.values())
        assert len(t['time'])==201 and len(t['motor_rates'])==200
        np.testing.assert_allclose(np.diff(t['time']),.01,rtol=0,atol=1e-10)
        # Runtime checks afferent membership against authenticated model rows;
        # this independent reconstruction checks identities against annotations.
        sensor=NamedLegPosition(sensory,scene,[n['cns_row'] for n in sensory['neurons']],motor_rows,r['sensory_polarity'])
        assert sensor.groups==r['sensory_groups'] and sensor.missing==r['missing_sensory_groups']
        drive=AggregateLegMuscles(motor,motor_rows,scene,r['polarity'])
        for i,rates in enumerate(t['motor_rates']):
            q=t['qpos'][0 if mode=='frozen' else i]
            np.testing.assert_array_equal(sensor.offsets(q),t['named_sensory_offsets'][i])
            drive.step(rates,t['qpos'][i],t['qvel'][i],zero_motor=mode=='zero')
            np.testing.assert_array_equal(muscle_coefficients(drive),t['muscle_coefficients'][i])
        if mode=='zero':
            np.testing.assert_array_equal(t['actuator_force'][1:],0)
        unsupported=[b['actuators'][i] for b in scene['bodies'] for i in range(90) if i not in {g['channel'] for g in drive.groups}]
        np.testing.assert_array_equal(t['actuator_force'][1:,unsupported],0)
        separation=np.stack([t['motor_rates'][:,g['cohorts'][0],:].mean(1)-t['motor_rates'][:,g['cohorts'][1],:].mean(1) for g in drive.groups])
        reports[mode]=dict(simulated_seconds=float(t['time'][-1]),
            named_neurons=sum(len(g['rows']) for g in sensor.groups),missing=sensor.missing,
            maximum_antagonist_rate_separation=float(np.abs(separation).max()),
            final_up=t['thorax_up'][-1].tolist(),
            late_horizontal_path_mm=np.linalg.norm(np.diff(t['thorax_position'][100:,:,:2],axis=0),axis=2).sum(0).tolist())
        traces[mode]=t
    with np.load(root/'evidence/trials/physics-muscle-none-zero-001/trace.npz',allow_pickle=False) as t:
        np.testing.assert_array_equal(traces['zero']['qpos'],t['qpos'])
    report=dict(checked=True,verifier_sha256=sha(Path(__file__)),results=reports,
        plus_vs_frozen_max_root_difference_mm=float(np.linalg.norm(traces['plus']['thorax_position']-traces['frozen']['thorax_position'],axis=2).max()),
        caveat='Transforms and force boundaries verified; trajectories and motor differences are not evidence of useful behavior.')
    with a.output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
