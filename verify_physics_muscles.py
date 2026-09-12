"""Verify recorded neural recruitment, coefficients and unpowered boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from aggregate_leg_muscles import AggregateLegMuscles
from physics_muscles import muscle_coefficients


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).parent;trials=root/'evidence/trials'
    scene=json.loads(a.source_scene.read_text())
    variant=json.loads((root/'evidence/passive-variant-none-001.json').read_text())
    assert hashlib.sha256(a.source_scene.read_bytes()).hexdigest()==variant['source_scene_sha256']
    annotation=json.loads((root/'motor-annotations.json').read_text())
    reference=json.loads((trials/'proprioceptor-circuit-004/result.json').read_text())
    with np.load(trials/'proprioceptor-circuit-004/trace.npz',allow_pickle=False) as archive:
        motor_rows=archive['motor_rows'].copy()
    results={};traces={}
    for name in ('physics-muscle-none-driven-001','physics-muscle-none-zero-001','physics-muscle-none-frozen-001'):
        r=json.loads((trials/name/'result.json').read_text())
        assert r['completed'] and r['scene_sha256']==variant['derived_scene_sha256']
        assert r['model_manifest_sha256']==reference['model_manifest_sha256']
        for key,file in (('runner_sha256','run_physics_muscles.py'),('muscle_code_sha256','aggregate_leg_muscles.py'),
                         ('force_adapter_sha256','physics_muscles.py'),('motor_annotation_sha256','motor-annotations.json')):
            assert r[key]==hashlib.sha256((root/file).read_bytes()).hexdigest()
        with np.load(trials/name/'trace.npz',allow_pickle=False) as archive:
            t={k:archive[k].copy() for k in archive.files}
        assert all(np.isfinite(v).all() for v in t.values())
        assert len(t['time'])==r['steps']+1 and len(t['motor_rates'])==r['steps']
        np.testing.assert_allclose(np.diff(t['time']),.01,rtol=0,atol=1e-10)
        d=AggregateLegMuscles(annotation,motor_rows,scene,r['polarity'])
        maximum=0.
        for i,rates in enumerate(t['motor_rates']):
            expected=d.step(rates,t['qpos'][i],t['qvel'][i],zero_motor=r['condition']=='zero-motor')
            np.testing.assert_allclose(expected,t['nominal_held_command'][i],rtol=0,atol=1e-7)
            difference=float(np.max(np.abs(muscle_coefficients(d)-t['muscle_coefficients'][i])))
            maximum=max(maximum,difference)
            assert difference<=1e-7
        channels={g['channel'] for g in d.groups}
        unsupported=[body['actuators'][i] for body in scene['bodies'] for i in range(90) if i not in channels]
        np.testing.assert_array_equal(t['actuator_force'][1:,unsupported],0)
        if r['condition']=='zero-motor':
            np.testing.assert_array_equal(t['muscle_coefficients'],0)
            np.testing.assert_array_equal(t['actuator_force'][1:],0)
        results[name]=dict(simulated_seconds=float(t['time'][-1]),
                           coefficient_recomputation_max_error=maximum,
                           maximum_recorded_joint_or_root_speed=float(np.abs(t['qvel']).max()),
                           wall_seconds=r['wall_seconds'])
        traces[name]=t
    old=np.load(trials/'passive-none-zero-001/trace.npz',allow_pickle=False)
    zero=traces['physics-muscle-none-zero-001']
    results['zero_mode_regression_max_qpos_difference']=float(np.abs(zero['qpos']-old['qpos'][:len(zero['qpos'])]).max())
    driven=traces['physics-muscle-none-driven-001'];frozen=traces['physics-muscle-none-frozen-001']
    n=len(frozen['time'])
    np.testing.assert_array_equal(driven['qpos'][0],frozen['qpos'][0])
    results['driven_vs_frozen_max_thorax_difference_mm']=float(np.linalg.norm(driven['thorax_position'][:n]-frozen['thorax_position'],axis=2).max())
    report=dict(checked=True,results=results,
                caveat='Checks transforms, timing and force boundaries, not biological competence or timestep convergence.',
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
