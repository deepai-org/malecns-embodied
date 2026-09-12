"""Check replicated tendon lengths and moment arms away from the reference pose."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco


def moments(model,data):
    matrix=np.zeros((model.nu,model.nv))
    for i in range(model.nu):
        start=int(data.moment_rowadr[i]);end=start+int(data.moment_rownnz[i])
        matrix[i,data.moment_colind[start:end]]=data.actuator_moment[start:end]
    return matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-xml','derived-xml','receipt','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=json.loads(a.receipt.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(a.source_xml)==r['source_xml_sha256'] and sha(a.derived_xml)==r['xml_sha256']
    source=mujoco.MjModel.from_xml_path(str(a.source_xml));target=mujoco.MjModel.from_xml_path(str(a.derived_xml))
    sd=mujoco.MjData(source);td=mujoco.MjData(target)
    mujoco.mj_resetDataKeyframe(source,sd,0);mujoco.mj_resetDataKeyframe(target,td,0)
    q0=sd.qpos.copy();qt0=td.qpos.copy();names=[source.joint(i).name for i in range(source.njnt) if source.joint(i).name.startswith('joint_LF')]
    sj=[source.joint(n).id for n in names];sq=[int(source.jnt_qposadr[j]) for j in sj];sv=[int(source.jnt_dofadr[j]) for j in sj]
    cases=[]
    for axis in range(7):
        for sign in (-1,1):
            sd.qpos[:]=q0;sd.qpos[sq[axis]]+=sign*.02;sd.qvel[:]=0;mujoco.mj_forward(source,sd)
            td.qpos[:]=qt0;td.qvel[:]=0
            for tag in ('lf','lm','lh','rf','rm','rh'):
                for n,q in zip(names,sq):td.qpos[target.jnt_qposadr[target.joint(tag+'/'+n).id]]=sd.qpos[q]
            mujoco.mj_forward(target,td);sm=moments(source,sd);tm=moments(target,td)
            length_error=moment_error=0.
            for tag in ('lf','lm','lh','rf','rm','rh'):
                tv=[int(target.jnt_dofadr[target.joint(tag+'/'+n).id]) for n in names]
                for i in range(source.nu):
                    j=target.actuator(tag+'/'+source.actuator(i).name).id
                    moment_error=max(moment_error,float(np.abs(sm[i,sv]-tm[j,tv]).max()))
                    st=int(source.actuator_trnid[i,0]);tt=int(target.actuator_trnid[j,0])
                    length_error=max(length_error,abs(float(sd.ten_length[st]-td.ten_length[tt])))
            assert length_error<1e-8 and moment_error<1e-8
            cases.append(dict(axis=names[axis],delta_radians=sign*.02,maximum_length_error=length_error,maximum_moment_error=moment_error))
    result=dict(checked=True,runner_sha256=sha(Path(__file__)),source_xml_sha256=sha(a.source_xml),derived_xml_sha256=sha(a.derived_xml),
        cases=cases,caveat='Checks geometric replication, not measured homology, force calibration, contact mechanics or behavior.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checked=True,cases=len(cases),maximum_length_error=max(c['maximum_length_error'] for c in cases),maximum_moment_error=max(c['maximum_moment_error'] for c in cases)),indent=2))


if __name__=='__main__':main()
