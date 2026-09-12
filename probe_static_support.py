"""Static support feasibility at recorded poses; never a runtime controller.

Non-leg joints are assumed ideally locked. Contact margins and fixed contact
locations are retained. This is a pose-specific mechanical relaxation, not
proof that the actual fly can stand or that every other pose is infeasible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def contact_rays(model,data,contact):
    jac=np.zeros((6,model.nv))
    for sign,geom in [(-1,int(contact.geom[0])),(1,int(contact.geom[1]))]:
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jp,jr,contact.pos,int(model.geom_bodyid[geom]))
        jac[:3]+=sign*jp; jac[3:]+=sign*jr
    frame=contact.frame.reshape(3,3)
    normal=np.concatenate([frame[0],np.zeros(3)])
    directions=[np.concatenate([frame[1],np.zeros(3)]),np.concatenate([frame[2],np.zeros(3)]),
                np.concatenate([np.zeros(3),frame[0]]),np.concatenate([np.zeros(3),frame[1]]),
                np.concatenate([np.zeros(3),frame[2]])]
    rays=[]
    for i in range(int(contact.dim)-1):
        for sign in (-1,1):rays.append(jac.T@(normal+sign*contact.friction[i]*directions[i]))
    return rays or [jac.T@normal]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','trial','muscle-receipt','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text())
    receipt=json.loads((a.trial/'result.json').read_text());muscles=json.loads(a.muscle_receipt.read_text())
    assert receipt['completed'] and receipt['scene_sha256']==sha(a.scene)
    xml=a.scene.parent/scene['scene_xml'];assert sha(xml)==scene['scene_xml_sha256']
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    with np.load(a.trial/'trace.npz',allow_pickle=False) as t:
        poses=t['qpos'][[100,200]].copy();times=t['time'][[100,200]].copy()
    results=[]
    for q,time in zip(poses,times):
        data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data)
        for resident in scene['residents']:
            prefix=resident['id']+'/'
            root=int(model.jnt_dofadr[resident['free_joint_id']])
            joints={j['semantic_id']:j for j in resident['joint_dofs126']}
            actuators=resident['actuators90'][:42]
            assert all(any(tag+'_coxa' in x['semantic_id'] or x['semantic_id'].startswith(tag+'_') for tag in ('lf','lm','lh','rf','rm','rh')) for x in actuators)
            dofs=list(range(root,root+6))+[joints[x['semantic_id'].removesuffix('-position')]['dof_address'] for x in actuators]
            assert len(set(dofs))==48
            rays=[];contacts=[]
            for c in data.contact:
                names=[model.geom(int(g)).name for g in c.geom]
                if not any(n.startswith(prefix) for n in names):continue
                # Include only ground/ecology contacts, not other fly/self contact.
                if not any(n.startswith('ecology/') for n in names):continue
                if c.dist>c.includemargin:continue
                rays.extend([v[dofs] for v in contact_rays(model,data,c)])
                contacts.append(dict(geoms=names,dist=float(c.dist),margin=float(c.includemargin),dim=int(c.dim)))
            contact=np.stack(rays,axis=1) if rays else np.empty((48,0))
            gravity=data.qfrc_bias[dofs].copy(); scale=max(1.,float(np.abs(gravity).max()))
            cases={}
            for name,channels in [('none',[]),('current18',[g['channel'] for g in muscles['muscle_groups']]),('current18_unbounded',[g['channel'] for g in muscles['muscle_groups']]),('all42',list(range(42)))]:
                assert all(0<=c<42 for c in channels)
                motor=np.zeros((48,len(channels)));bounds=[(0,None)]*len(rays)
                for col,ch in enumerate(channels):
                    motor[6+ch,col]=1
                    aid=actuators[ch]['actuator_id'];assert np.allclose(model.actuator_gear[aid],[1,0,0,0,0,0])
                    bounds.append((None,None) if name=='current18_unbounded' else tuple(model.actuator_forcerange[aid]))
                matrix=np.column_stack([contact,motor])
                solution=linprog(np.concatenate([np.ones(len(rays)),np.zeros(len(channels))]),
                    A_eq=matrix/scale,b_eq=gravity/scale,bounds=bounds,method='highs') if matrix.shape[1] else None
                feasible=bool(solution is not None and solution.success)
                cases[name]=dict(feasible=feasible,status=None if solution is None else int(solution.status),
                    solver_message=None if solution is None else solution.message,
                    maximum_balance_residual=None if not feasible else float(np.abs(matrix@solution.x-gravity).max()),
                    maximum_motor_torque=None if not feasible or not channels else float(np.abs(solution.x[len(rays):]).max()))
            # A diagnostic lower bound on absent joint reactions: known motor
            # torques are unlimited, while absolute unsupported torques are
            # minimized. These torques are never sent to the physical host.
            known={g['channel'] for g in muscles['muscle_groups']}
            missing=[i for i in range(42) if i not in known]
            matrix=np.column_stack([contact,np.vstack([np.zeros((6,42)),np.eye(42)]),np.zeros((48,len(missing)))])
            upper=np.zeros((2*len(missing),matrix.shape[1]))
            for i,ch in enumerate(missing):
                upper[2*i,len(rays)+ch]=1;upper[2*i+1,len(rays)+ch]=-1
                upper[2*i:2*i+2,len(rays)+42+i]=-1
            fit=linprog(np.concatenate([np.zeros(len(rays)+42),np.ones(len(missing))]),
                A_eq=matrix/scale,b_eq=gravity/scale,A_ub=upper,b_ub=np.zeros(len(upper)),
                bounds=[(0,None)]*len(rays)+[(None,None)]*42+[(0,None)]*len(missing),method='highs')
            deficit=dict(status=int(fit.status),solver_message=fit.message,
                minimum_total_missing_torque=None if not fit.success else float(fit.fun),
                missing_joint_torques=None if not fit.success else {actuators[ch]['semantic_id']:float(fit.x[len(rays)+ch]) for ch in missing},
                maximum_balance_residual=None if not fit.success else float(np.abs(matrix@fit.x-gravity).max()))
            results.append(dict(time=float(time),resident=resident['id'],contacts=contacts,cases=cases,deficit=deficit))
    report=dict(kind='pose-specific static support relaxation',runner_sha256=sha(Path(__file__)),
        scene_sha256=sha(a.scene),trace_sha256=sha(a.trial/'trace.npz'),muscle_receipt_sha256=sha(a.muscle_receipt),
        mujoco=mujoco.__version__,results=results,
        assumptions=['Recorded pose frozen at zero velocity; no CNS or controller runs.',
                     'Gravity balance on root and 42 leg actuator coordinates; other joints assumed ideally locked.',
                     'No passive spring forces; actuators may choose arbitrary signed torque up to original capacity, or unbounded in the explicitly named case.',
                     'Fixed generated ground/ecology contact locations, including positive margins; pyramidal friction rays.',
                     'Feasibility is not behavioral competence; infeasibility is limited to these poses and assumptions.'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='contacts'} for r in results],indent=2))


if __name__=='__main__':main()
