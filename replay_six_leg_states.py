"""Replay selected physical intervals from complete saved MuJoCo states."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('xml','trial','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=json.loads((a.trial/'result.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert r['completed'] and sha(a.xml)==r['input_sha256']['xml']
    model=mujoco.MjModel.from_xml_path(str(a.xml));spec=mujoco.mjtState.mjSTATE_INTEGRATION
    assert int(spec)==r['physics_state_spec'] and mujoco.mj_stateSize(model,spec)==r['physics_state_size']
    samples=[];contacts=[]
    with np.load(a.trial/'trace.npz',allow_pickle=False) as t:
        for tick in (0,25,100,175,199):
            for lane in range(6):
                data=mujoco.MjData(model);mujoco.mj_setState(model,data,t['integration_state'][tick,lane],spec)
                data.ctrl[:]=t['controls'][tick,lane]
                for _ in range(round(.01/model.opt.timestep)):mujoco.mj_step(model,data)
                actual=np.empty(r['physics_state_size']);mujoco.mj_getState(model,data,actual,spec)
                error=float(np.abs(actual-t['integration_state'][tick+1,lane]).max())
                assert error<1e-7
                samples.append(dict(tick=tick,lane=lane,maximum_state_error=error))
        for lane in range(6):
            data=mujoco.MjData(model);mujoco.mj_setState(model,data,t['integration_state'][-1,lane],spec);mujoco.mj_forward(model,data)
            forces={}
            for contact_id,contact in enumerate(data.contact):
                names=[model.geom(int(g)).name for g in contact.geom]
                if 'floor' not in names:continue
                index=next(i for i,n in enumerate(names) if n!='floor');name=names[index]
                force=np.zeros(6);mujoco.mj_contactForce(model,data,contact_id,force)
                forces[name]=forces.get(name,0.)+float(force[0])
            contacts.append(dict(lane=lane,normal_force_by_contacting_geom=forces))
    result=dict(checked=True,runner_sha256=sha(Path(__file__)),trace_sha256=sha(a.trial/'trace.npz'),
        xml_sha256=sha(a.xml),samples=samples,final_contacts=contacts,maximum_state_error=max(s['maximum_state_error'] for s in samples),
        caveat='Replays physical intervals with recorded controls; not independent neural regeneration or evidence of useful behavior.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checked=True,intervals=len(samples),maximum_state_error=result['maximum_state_error']),indent=2))


if __name__=='__main__':main()
