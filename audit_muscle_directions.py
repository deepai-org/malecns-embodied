"""Static actuator torque directions in the published foreleg geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('xml','original-mapping','pooled-mapping','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); old=json.loads(a.original_mapping.read_text());new=json.loads(a.pooled_mapping.read_text())
    digest=hashlib.sha256(a.xml.read_bytes()).hexdigest()
    assert digest==old['xml_sha256']==new['xml_sha256']
    model=mujoco.MjModel.from_xml_path(str(a.xml));data=mujoco.MjData(model)
    assert model.nu==model.na==15
    mujoco.mj_resetDataKeyframe(model,data,0);data.qvel[:]=0;data.ctrl[:]=0;data.act[:]=0
    mujoco.mj_forward(model,data);baseline=data.qfrc_actuator.copy()
    torque=[]
    for muscle in range(model.nu):
        data.act[:]=0;data.act[muscle]=1;data.ctrl[:]=data.act;mujoco.mj_forward(model,data)
        torque.append(data.qfrc_actuator.copy()-baseline)
    torque=np.asarray(torque).T
    left=[int(model.jnt_dofadr[i]) for i in range(model.njnt) if model.joint(i).name.startswith('joint_LF')]
    assert len(left)==7
    conditions={}
    for name,mapping in [('original_six',old),('pooled_fifteen',new),('pooled_shared_neurons',new)]:
        chosen=[m['muscle_id'] for m in mapping['muscles'] if m['candidate_motor_rows']]
        matrix=torque[np.ix_(left,chosen)]
        if name=='pooled_shared_neurons':
            rows=sorted({r for m in mapping['muscles'] for r in m['candidate_motor_rows']})
            weights=np.zeros((len(chosen),len(rows)))
            for m in mapping['muscles']:
                weights[chosen.index(m['muscle_id']),[rows.index(r) for r in m['candidate_motor_rows']]]=1/len(m['candidate_motor_rows'])
            matrix=matrix@weights
        directions={}
        for joint in range(7):
            for sign in (-1,1):
                target=np.eye(7)[:,joint]*sign
                # Unit torque target with nonnegative, unbounded muscle drive:
                # tests cone coverage, not physiological force feasibility.
                fit=linprog(np.ones(matrix.shape[1]),A_eq=matrix,b_eq=target,bounds=(0,None),method='highs')
                directions[f'{joint}:{sign}']=dict(status=int(fit.status),feasible=bool(fit.success),
                    residual=None if not fit.success else float(np.abs(matrix@fit.x-target).max()))
        conditions[name]=dict(actuators=chosen,independent_variables=matrix.shape[1],rank=int(np.linalg.matrix_rank(matrix)),singular_values=np.linalg.svd(matrix,compute_uv=False).tolist(),directions=directions)
    report=dict(kind='foreleg muscle authority at fixed keyframe; not a behavioral trial',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),xml_sha256=digest,
        mapping_sha256={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('original_mapping','pooled_mapping')},
        mujoco=mujoco.__version__,joint_names=[model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith('joint_LF')],
        actuator_names=[model.actuator(i).name for i in range(model.nu)],
        active_torque_per_unit_activation=torque[left].tolist(),passive_actuator_torque=baseline[left].tolist(),
        conditions=conditions,
        caveats=['Static zero-velocity keyframe; no motor neurons run.',
                 'Cone test allows activation above one; feasibility establishes direction coverage only.',
                 'First two cases vary actuators independently; third includes shared-pool recruitment but not CNS trajectory constraints.',
                 'Published geometry/parameters are approximate and foreleg-only; no whole-body transfer is established.'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:dict(rank=v['rank'],directions_feasible=sum(x['feasible'] for x in v['directions'].values())) for k,v in conditions.items()},indent=2))


if __name__=='__main__':main()
