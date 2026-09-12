"""Verify supplied pool recruitment and position transduction in saved trials."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from neuromuscular import NamedMuscleDrive
from run_pooled_feedback import position_current


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).parent;trials=root/'evidence/trials'
    mapping=json.loads((root/'evidence/pooled-muscle-mapping-001.json').read_text())
    with np.load(trials/'proprioceptor-circuit-004/trace.npz',allow_pickle=False) as t: rows=t['motor_rows'].copy()
    reference=json.loads((trials/'proprioceptor-circuit-004/result.json').read_text())
    drive=NamedMuscleDrive(mapping,rows);results={}
    for name,runner in [('pooled-feedback-001','run_muscle_feedback.py'),('pooled-feedback-002','run_pooled_feedback.py')]:
        r=json.loads((trials/name/'result.json').read_text())
        assert r['completed'] and r['model_manifest_sha256']==reference['model_manifest_sha256']
        assert r['runner_sha256']==sha(root/runner)
        assert r['input_sha256']['mapping']==sha(root/'evidence/pooled-muscle-mapping-001.json')
        assert r['input_sha256']['xml']==mapping['xml_sha256']
        assert r['input_sha256']['sensory']==sha(root/'proprioceptor-annotations.json')
        with np.load(trials/name/'trace.npz',allow_pickle=False) as archive:t={k:archive[k].copy() for k in archive.files}
        assert all(np.isfinite(v).all() for v in t.values())
        assert t['controls'].shape==(200,6,15) and t['motor_rates'].shape==(200,815,6)
        maximum=0.
        for i in range(200):
            for lane,(condition,polarity) in enumerate(r['conditions']):
                expected=np.zeros(15) if condition=='zero-motor' else drive(t['motor_rates'][i,:,lane])
                maximum=max(maximum,float(np.abs(expected-t['controls'][i,lane]).max()))
                np.testing.assert_allclose(expected,t['controls'][i,lane],rtol=0,atol=6e-8)
                if condition=='zero-motor':np.testing.assert_array_equal(t['controls'][i,lane],0)
                # Pinned source model's left tibia pitch is qpos index 6.
                q=t['qpos'][0 if condition=='frozen-sensory' else i,lane,6]
                np.testing.assert_allclose(position_current(q,r['joint_range_radians'],r['amplitude_model_units'],polarity),t['sensory_current'][i,lane],rtol=0,atol=1e-12)
        results[name]=dict(maximum_control_reconstruction_error=maximum,simulated_seconds=r['simulated_seconds'],
            late_feedback_joint_range_rad=float(np.ptp(t['qpos'][-100:,[0,3]],axis=0).max()))
    geometry=json.loads((root/'evidence/muscle-directions-001.json').read_text())
    assert geometry['runner_sha256']==sha(root/'audit_muscle_directions.py')
    assert geometry['mapping_sha256']['pooled_mapping']==sha(root/'evidence/pooled-muscle-mapping-001.json')
    matrix=np.asarray(geometry['active_torque_per_unit_activation'])
    assert np.linalg.matrix_rank(matrix)==geometry['conditions']['pooled_fifteen']['rank']==7
    result=dict(checked=True,verifier_sha256=sha(Path(__file__)),trials=results,
        caveat='Recomputes transforms and raw matrix rank, not biological competence or all LP direction results.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
