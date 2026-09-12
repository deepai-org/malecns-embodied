"""Choose a fixed rig axis for approximate named coxa protraction/retraction.

Geometry-based coordinate mapping, not a behavioral controller or measured
muscle moment arm. Evaluated once at the original rig keyframe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text())
    xml=a.scene.parent/scene['scene_xml']
    assert hashlib.sha256(xml.read_bytes()).hexdigest()==scene['scene_xml_sha256']
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    q0=data.qpos.copy(); resident=scene['residents'][0]; prefix=resident['id']+'/'
    thorax=model.body(prefix+'c_thorax').id
    head=model.body(prefix+'c_head').id
    forward=data.xpos[head]-data.xpos[thorax];forward/=np.linalg.norm(forward)
    groups=[];eps=1e-4
    for tag in ('lf','lm','lh','rf','rm','rh'):
        endpoint=model.body(prefix+tag+'_tibia').id
        candidates=[]
        for axis in ('yaw','pitch','roll'):
            joint=f'c_thorax-{tag}_coxa-{axis}'
            entries=[j for j in resident['joint_dofs126'] if j['semantic_id']==joint]
            assert len(entries)==1
            q=entries[0]['qpos_address']; values=[]
            for delta in (-eps,eps):
                data.qpos[:]=q0;data.qpos[q]+=delta;mujoco.mj_forward(model,data)
                values.append(float(np.dot(data.xpos[endpoint]-data.xpos[thorax],forward)))
            candidates.append(dict(joint=joint,forward_derivative_mm_per_rad=(values[1]-values[0])/(2*eps)))
        chosen=max(candidates,key=lambda x:abs(x['forward_derivative_mm_per_rad']))
        assert abs(chosen['forward_derivative_mm_per_rad'])>1e-3
        groups.append(dict(leg=tag,joint=chosen['joint'],positive_protraction_sign=1 if chosen['forward_derivative_mm_per_rad']>0 else -1,candidates=candidates))
    result=dict(kind='fixed one-axis coxa approximation at rig rest',groups=groups,
        scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        scene_xml_sha256=scene['scene_xml_sha256'],mujoco=mujoco.__version__,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forward_definition='unit vector thorax origin to head origin at original keyframe',
        epsilon_radians=eps,
        caveats=['One selected coordinate per coxa, not anatomical muscle insertion geometry.',
                 'Local protraction sign at rig rest need not hold over every posture.',
                 'No online axis selection, body-pose feedback controller or target trajectory.'])
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
