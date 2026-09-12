"""Audit initial intersections and passive collision response; no CNS runs.

Collision disabling is a diagnostic only and never edits the scene on disk.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contacts(model, data):
    result = []
    for c in data.contact:
        names = [model.geom(int(g)).name for g in c.geom]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, len(result), force)
        result.append(dict(geoms=names, distance=float(c.dist),
                           normal_force=float(force[0]),
                           active=bool(c.efc_address >= 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('xml', 'body-receipt', 'source-xml', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.body_receipt.read_text())
    assert sha(args.xml) == receipt['xml_sha256']
    assert sha(args.source_xml) == 'd67ff06d0c684599f00d5f33f8f8ac8ced01ab994c41341b551f8148b6d8aa0b'
    source = mujoco.MjModel.from_xml_path(str(args.source_xml))
    source_data = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, source_data, 0)
    mujoco.mj_forward(source, source_data)
    cases = []
    for gravity in ('original', 'zero'):
        for collision in ('original', 'ground-only'):
            model = mujoco.MjModel.from_xml_path(str(args.xml))
            assert model.nv == 48 and model.nu == model.na == 90
            if gravity == 'zero':
                model.opt.gravity[:] = 0
            if collision == 'ground-only':
                # A collision pair needs (type1 & affinity2) or its converse.
                # Keep floor-body interactions; remove all body-body pairs.
                assert model.npair == 0
                model.geom_contype[:] = 2
                model.geom_conaffinity[:] = 1
                floor = model.geom('floor').id
                model.geom_contype[floor] = 1
                model.geom_conaffinity[floor] = 2
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.ctrl[:] = 0
            data.act[:] = 0
            initial = data.qpos.copy()
            mujoco.mj_forward(model, data)
            initial_contacts = contacts(model, data)
            initial_acceleration = data.qacc.copy()
            snapshots = []
            for step in range(501):
                if step % 50 == 0:
                    mujoco.mj_forward(model, data)
                    mujoco.mj_energyVel(model, data)
                    snapshots.append(dict(time=float(data.time),
                        qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                        kinetic_energy=float(data.energy[1]),
                        max_joint_change=float(np.abs(data.qpos[7:] - initial[7:]).max()),
                        contacts=contacts(model, data)))
                if step < 500:
                    mujoco.mj_step(model, data)
                    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            assert abs(data.time - 500 * model.opt.timestep) < 1e-10
            if collision == 'ground-only':
                assert all('floor' in c['geoms'] for s in snapshots for c in s['contacts'])
            cases.append(dict(gravity=gravity, collision=collision,
                initial_contacts=initial_contacts,
                initial_acceleration=initial_acceleration.tolist(), snapshots=snapshots))
    report = dict(kind='initialization and passive collision diagnostic, not behavior',
        runner_sha256=sha(Path(__file__)), mujoco=mujoco.__version__,
        input_sha256={name: sha(getattr(args, name)) for name in ('xml', 'body_receipt', 'source_xml')},
        source_initial_contacts=contacts(source, source_data), cases=cases,
        assumptions=[
            'Original keyframe, zero commands and zero initial activation in every case.',
            'Passive muscles and subsequent minimum activation remain; this is not paralysis.',
            'Only gravity or collision masks change in memory; source files are untouched.',
            'Ground-only collision is an ablation, not an accepted anatomical repair.',
            'Short finite physics runs cannot establish support, CNS function or useful behavior.'])
    with args.output.open('x') as out:
        json.dump(report, out, indent=2)
    for case in cases:
        print(json.dumps(dict(gravity=case['gravity'], collision=case['collision'],
            initial_contacts=len(case['initial_contacts']),
            minimum_distance=min((c['distance'] for c in case['initial_contacts']), default=None),
            initial_peak_acceleration=max(map(abs, case['initial_acceleration'])),
            final_kinetic_energy=case['snapshots'][-1]['kinetic_energy'],
            final_max_joint_change=case['snapshots'][-1]['max_joint_change'])))


if __name__ == '__main__':
    main()
