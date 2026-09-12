"""Offline collision/foot-placement search, never a runtime controller."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('xml', 'body-receipt', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--refine', type=Path, help='Prior search receipt; refine its poses without pose regularization')
    a = p.parse_args()
    receipt = json.loads(a.body_receipt.read_text())
    assert sha(a.xml) == receipt['xml_sha256']
    m = mujoco.MjModel.from_xml_path(str(a.xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    initial = d.qpos.copy()
    assert m.nq == 49 and m.nv == 48 and m.nu == 90
    tags = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
    feet = [m.geom(tag + '/LFTarsus5_geom').id for tag in tags]
    vertices = []
    for g in feet:
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        vertices.append(m.mesh_vert[start:start + count].copy())
    joint_ids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert [int(m.jnt_qposadr[j]) for j in joint_ids] == list(range(7, 49))
    # Several source ranges are declared but not enforced by the simulator.
    # Use all declared ranges as conservative *search* bounds, not new physics.
    assert (m.jnt_range[joint_ids, 1] > m.jnt_range[joint_ids, 0]).all()
    lower = np.r_[.9, m.jnt_range[joint_ids, 0]]
    upper = np.r_[2.3, m.jnt_range[joint_ids, 1]]
    pairs = [(i, j) for i in range(m.ngeom) for j in range(i + 1, m.ngeom)]
    index = {pair: k for k, pair in enumerate(pairs)}

    def evaluate(x):
        d.qpos[:] = initial
        d.qpos[2] = x[0]
        d.qpos[7:] = x[1:]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        penetration = np.zeros(len(pairs))
        for c in d.contact:
            key = tuple(sorted(map(int, c.geom)))
            penetration[index[key]] = max(penetration[index[key]], -float(c.dist))
        bottom = [float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
                  for g, v in zip(feet, vertices)]
        return penetration, np.asarray(bottom), d.geom_xpos[feet, :2].copy()

    _, _, target_xy = evaluate(np.r_[1.7, initial[7:]])
    began = time.monotonic()
    results = []
    if a.refine:
        previous = json.loads(a.refine.read_text())
        assert previous['xml_sha256'] == sha(a.xml)
        starts = [np.r_[r['qpos'][2], r['qpos'][7:]] for r in previous['results']]
    else:
        starts = [np.r_[height, initial[7:]] for height in (1.3, 1.7, 2.1)]
    # Fixed starts, not a search over model parameters or neural gains.
    for start in starts:
        height = float(start[0])
        start = np.clip(start, lower + 1e-8, upper - 1e-8)

        def objective(x):
            penetration, bottom, xy = evaluate(x)
            if a.refine:
                return np.r_[10 * penetration, 10 * bottom]
            return np.r_[10 * penetration, 10 * bottom,
                         .01 * (xy - target_xy).ravel(), .001 * (x[1:] - initial[7:])]

        fit = least_squares(objective, start, bounds=(lower, upper), max_nfev=120,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8, tr_solver='lsmr')
        penetration, bottom, xy = evaluate(fit.x)
        details = [dict(geoms=[m.geom(i).name for i in pair], penetration=float(value))
                   for pair, value in zip(pairs, penetration) if value > 1e-6]
        # Candidate geometry only. Neither equilibrium nor force capacity tested.
        candidate = bool(penetration.max() < 1e-5 and abs(bottom).max() < 1e-4)
        result = dict(start_height=height, status=int(fit.status), message=fit.message,
                      nfev=int(fit.nfev), cost=float(fit.cost), qpos=d.qpos.tolist(),
                      maximum_penetration=float(penetration.max()), foot_bottom_z=bottom.tolist(),
                      foot_xy=xy.tolist(), penetrations=details, geometric_candidate=candidate)
        results.append(result)
        print(json.dumps({k: result[k] for k in ('start_height', 'nfev', 'maximum_penetration',
                                               'foot_bottom_z', 'geometric_candidate')}), flush=True)
    report = dict(kind='offline initial-pose search; not standing or behavior',
        runner_sha256=sha(Path(__file__)), xml_sha256=sha(a.xml),
        refine_sha256=sha(a.refine) if a.refine else None,
        body_receipt_sha256=sha(a.body_receipt), mujoco=mujoco.__version__,
        wall_seconds=time.monotonic() - began, results=results,
        search_joint_ranges=m.jnt_range[joint_ids].tolist(),
        runtime_joint_limited=m.jnt_limited[joint_ids].tolist(),
        assumptions=['Body geometry, collision masks, joint limits and muscle parameters unchanged.',
                     'Search respects all declared source joint ranges, including those not runtime-enforced.',
                     'Only initial joint coordinates and root height vary; root orientation stays fixed.',
                     'Contact penetration uses compiled engine collision geometry; feet use mesh minimum height.',
                     'Geometric candidate thresholds: penetration <1e-5 and foot-height error <1e-4 model units.',
                     'No muscle/gravity equilibrium, CNS activity, runtime controller or behavior is tested.',
                     'Failure of this bounded local search is not proof that no valid pose exists.'])
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
