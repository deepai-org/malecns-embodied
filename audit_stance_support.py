"""Static muscle/contact balance at offline poses; never a controller."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog
from pooled_muscle_mapping import POOLS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('xml', 'poses', 'motor', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    poses = json.loads(a.poses.read_text())
    assert sha(a.xml) == poses['xml_sha256']
    m = mujoco.MjModel.from_xml_path(str(a.xml))
    annotations = json.loads(a.motor.read_text())['motors']
    weights = np.zeros((m.nu, len(annotations)))
    for i in range(m.nu):
        tag, name = m.actuator(i).name.split('/', 1)
        selected = [j for j, r in enumerate(annotations)
                    if r['mancType'] in POOLS[name]
                    and r['subclass'] == {'f': 'fl', 'm': 'ml', 'h': 'hl'}[tag[1]]
                    and (r['somaSide'] or r['rootSide']) == tag[0].upper()]
        if selected:
            weights[i, selected] = 1 / len(selected)
    weights = weights[:, np.any(weights, axis=0)]
    results = []
    for index, pose in enumerate(poses['results']):
        assert pose['geometric_candidate']
        d = mujoco.MjData(m)
        d.qpos[:] = pose['qpos']
        d.qvel[:] = 0
        d.act[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(m, d)
        worst = max((-float(c.dist) for c in d.contact), default=0.)
        assert worst < 1e-5
        baseline = d.qfrc_actuator.copy()
        passive = d.qfrc_passive.copy()
        target = d.qfrc_bias.copy() - baseline - passive
        columns = []
        for i in range(m.nu):
            d.act[:] = 0
            d.act[i] = 1
            mujoco.mj_forward(m, d)
            columns.append(d.qfrc_actuator.copy() - baseline)
        active = np.stack(columns, axis=1)
        d.act[:] = np.linspace(.1, .9, m.na)
        mujoco.mj_forward(m, d)
        affine_error = float(np.abs(d.qfrc_actuator - baseline - active @ d.act).max())
        assert affine_error < 1e-8, 'Actuation not affine: LP would be invalid'
        rays, supports = [], []
        for tag in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'):
            g = m.geom(tag + '/LFTarsus5_geom').id
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            world = m.mesh_vert[start:start + count] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            point = world[np.argmin(world[:, 2])].copy()
            assert abs(point[2]) < 1e-4
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, point, int(m.geom_bodyid[g]))
            friction = float(max(m.geom_friction[g, 0], m.geom_friction[m.geom('floor').id, 0]))
            for x, y in ((friction, 0), (-friction, 0), (0, friction), (0, -friction)):
                rays.append(jp.T @ np.array([x, y, 1.]))
            supports.append(dict(leg=tag, point=point.tolist(), friction=friction))
        contact = np.stack(rays, axis=1)
        conditions = {}
        for name, recruitment in (('passive', np.zeros((m.nu, 0))),
                                  ('independent90', np.eye(m.nu)), ('named_pools', weights),
                                  ('independent90_unbounded', np.eye(m.nu)),
                                  ('arbitrary_joint_torque', None)):
            torque = (np.vstack([np.zeros((6, 42)), np.eye(42)])
                      if recruitment is None else active @ recruitment)
            matrix = np.column_stack([contact, torque])
            motor_bound = ((None, None) if recruitment is None else
                           (0, None) if name.endswith('_unbounded') else (0, 1))
            fit = linprog(np.r_[np.ones(contact.shape[1]), np.zeros(torque.shape[1])],
                A_eq=matrix, b_eq=target,
                bounds=[(0, None)] * contact.shape[1] + [motor_bound] * torque.shape[1], method='highs')
            conditions[name] = dict(status=int(fit.status), message=fit.message,
                feasible=bool(fit.success),
                residual=None if not fit.success else float(np.abs(matrix @ fit.x - target).max()),
                activations=None if not fit.success or recruitment is None else (recruitment @ fit.x[contact.shape[1]:]).tolist(),
                arbitrary_joint_torques=None if not fit.success or recruitment is not None else fit.x[contact.shape[1]:].tolist(),
                support_ray_weights=None if not fit.success else fit.x[:contact.shape[1]].tolist())
        capacity = {}
        for name, recruitment in (('independent90', np.eye(m.nu)), ('named_pools', weights)):
            # Minimize the largest normalized input needed for static balance.
            # Extrapolation is diagnostic only; it never changes muscle forces.
            ncontact, ninput = contact.shape[1], recruitment.shape[1]
            matrix = np.column_stack([contact, active @ recruitment, np.zeros(m.nv)])
            upper = np.zeros((ninput, matrix.shape[1]))
            upper[:, ncontact:ncontact+ninput] = np.eye(ninput)
            upper[:, -1] = -1
            objective = np.r_[np.zeros(matrix.shape[1]-1), 1.]
            fit = linprog(objective, A_eq=matrix, b_eq=target,
                A_ub=upper, b_ub=np.zeros(ninput), bounds=(0, None), method='highs')
            activations = None if not fit.success else recruitment @ fit.x[ncontact:-1]
            capacity[name] = dict(status=int(fit.status), message=fit.message,
                minimum_peak_input=None if not fit.success else float(fit.fun),
                activations=None if activations is None else activations.tolist(),
                peak_muscles=[] if activations is None else
                    [m.actuator(i).name for i in range(m.nu) if activations[i] >= .99*fit.fun],
                maximum_balance_error=None if not fit.success else float(np.abs(matrix @ fit.x-target).max()),
                maximum_input_bound_violation=None if not fit.success else float(max(0., (upper @ fit.x).max())),
                dual_objective=None if not fit.success else float(target @ fit.eqlin.marginals),
                dual_stationarity_error=None if not fit.success else float(np.abs(
                    objective-matrix.T@fit.eqlin.marginals-upper.T@fit.ineqlin.marginals-fit.lower.marginals).max()))
        results.append(dict(pose_index=index, maximum_penetration=worst,
            affine_error=affine_error, supports=supports, conditions=conditions, capacity=capacity))
        print(json.dumps(dict(pose=index, capacity=capacity)), flush=True)
    report = dict(kind='pose-specific ideal foot-contact muscle equilibrium; not behavior',
        runner_sha256=sha(Path(__file__)),
        inputs={name: sha(getattr(a, name)) for name in ('xml', 'poses', 'motor')},
        pooling_sha256=sha(Path(__file__).with_name('pooled_muscle_mapping.py')),
        mujoco=mujoco.__version__, results=results,
        assumptions=['All 48 velocity coordinates balanced at zero velocity; passive forces retained.',
                     'One ideal contact at the lowest mesh vertex on each distal tarsus; no torso or self-contact support.',
                     'Conservative four-ray tangential friction cone, no adhesion or contact moments.',
                     'Ideal hard contacts; not proof of equilibrium under the runtime soft-contact solver.',
                     'Independent muscle or named-pool inputs may choose any value in [0,1]; no CNS reachability constraint.',
                     'Explicit unbounded-muscle case extrapolates the affine force law beyond physiological activation.',
                     'Arbitrary-joint-torque case tests contact/root support independent of muscle direction/capacity.',
                     'No forces or fitted activations are installed in a runtime controller.',
                     'Infeasibility concerns these poses and point-contact assumptions, not every possible stance.'])
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
