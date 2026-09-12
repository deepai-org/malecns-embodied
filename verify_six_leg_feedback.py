"""Reconstruct the free-body neural recruitment and proprioceptive transforms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pooled_muscle_mapping import POOLS
from run_pooled_feedback import position_current


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).parent;trial=root/'evidence/trials/six-leg-feedback-002'
    r=json.loads((trial/'result.json').read_text());body=json.loads((root/'evidence/six-leg-body-002.json').read_text())
    assert r['completed'] and r['runner_sha256']==sha(root/'run_six_leg_feedback.py')
    assert r['input_sha256']['body_receipt']==sha(root/'evidence/six-leg-body-002.json')
    assert body['floor_count']==1
    assert r['input_sha256']['xml']==body['xml_sha256']
    assert body['runner_sha256']==sha(root/'build_six_leg_muscles.py')
    assert r['pooling_code_sha256']==sha(root/'pooled_muscle_mapping.py')
    assert r['sensory_code_sha256']==sha(root/'run_pooled_feedback.py')
    for field,file in [('motor','motor-annotations.json'),('sensory','proprioceptor-annotations.json')]:assert r['input_sha256'][field]==sha(root/file)
    motor=json.loads((root/'motor-annotations.json').read_text());sensory=json.loads((root/'proprioceptor-annotations.json').read_text())
    reference=root/'evidence/trials/proprioceptor-circuit-004'
    assert r['model_manifest_sha256']==json.loads((reference/'result.json').read_text())['model_manifest_sha256']
    with np.load(reference/'trace.npz',allow_pickle=False) as t:rows=t['motor_rows'].copy()
    index={int(row):i for i,row in enumerate(rows)};weights=np.zeros((90,815),dtype=np.float32)
    for i,g in enumerate(r['recruitment']):
        tag,name=g['actuator'].split('/',1);labels=POOLS[name]
        selected=[m for m in motor['motors'] if m['mancType'] in labels and m['subclass']=={'f':'fl','m':'ml','h':'hl'}[tag[1]] and (m['somaSide'] or m['rootSide'])==tag[0].upper()]
        assert [m['cns_row'] for m in selected]==g['rows'] and [m['bodyId'] for m in selected]==g['body_ids']
        if selected:weights[i,[index[m['cns_row']] for m in selected]]=1/len(selected)
    for s in r['senses']:
        for kind,group in zip(('SNpp50','SNpp51'),s['groups']):
            tag=s['leg'];selected=[n['cns_row'] for n in sensory['neurons'] if n['type']==kind and n['entryNerve']=={'f':'ProLN','m':'MesoLN','h':'MetaLN'}[tag[1]] and (n['somaSide'] or n['rootSide'])==tag[0].upper()]
            assert selected==group
    with np.load(trial/'trace.npz',allow_pickle=False) as archive:t={k:archive[k].copy() for k in archive.files}
    assert all(np.isfinite(v).all() for v in t.values())
    assert t['qpos'].shape==(201,6,49) and t['controls'].shape==(200,6,90)
    assert t['act'].shape==(201,6,90) and t['integration_state'].shape==(201,6,r['physics_state_size'])
    np.testing.assert_allclose(np.diff(t['time']),.01,atol=1e-10,rtol=0)
    maximum=0.
    for tick in range(200):
        expected=(weights@t['motor_rates'][tick]).T
        for lane,(mode,polarity) in enumerate(r['conditions']):
            if mode=='zero-motor':expected[lane]=0
            for j,s in enumerate(r['senses']):
                q=t['qpos'][0 if mode=='frozen-sensory' else tick,lane,s['qpos']]
                np.testing.assert_allclose(position_current(q,s['limits'],.2,polarity),t['sensory_current'][tick,lane,j],atol=1e-12,rtol=0)
        maximum=max(maximum,float(np.abs(expected-t['controls'][tick]).max()))
        np.testing.assert_allclose(expected,t['controls'][tick],atol=6e-8,rtol=0)
    unassigned=[i for i,g in enumerate(r['recruitment']) if not g['rows']]
    np.testing.assert_array_equal(t['controls'][:,:,unassigned],0)
    for lane in (2,5):np.testing.assert_array_equal(t['controls'][:,lane],0)
    result=dict(checked=True,verifier_sha256=sha(Path(__file__)),maximum_control_reconstruction_error=maximum,
        unmapped_actuators=len(unassigned),final_up=t['thorax_up'][-1].tolist(),
        late_horizontal_path_mm=np.linalg.norm(np.diff(t['thorax_position'][100:,:,:2],axis=0),axis=2).sum(0).tolist(),
        feedback_vs_frozen_max_joint_difference_rad=[float(np.abs(t['qpos'][:,i,7:]-t['qpos'][:,i+1,7:]).max()) for i in (0,3)],
        zero_command_peak_actuator_force=float(np.abs(t['actuator_force'][:,[2,5]]).max()),
        caveat='Checks recorded transforms and state shapes, not behavioral usefulness or calibrated anatomy. Physical state replay is a separate audit.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
