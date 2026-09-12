"""Static tibia tip-equivalent force relative to model body weight.

Other joints ideally locked; not a reproduction of the biological force-probe
experiment, a calibration, or a runtime controller.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-xml', 'xml', 'poses', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert sha(a.source_xml) == 'd67ff06d0c684599f00d5f33f8f8ac8ced01ab994c41341b551f8148b6d8aa0b'
    poses = json.loads(a.poses.read_text())
    assert sha(a.xml) == poses['xml_sha256']
    results = []
    for kind, path in (('source', a.source_xml), ('replicated', a.xml)):
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        qposes = [d.qpos.copy()] if kind == 'source' else [r['qpos'] for r in poses['results']]
        for pose_index, q in enumerate(qposes):
            for prefix in (('',) if kind == 'source' else ('lf/', 'lm/', 'lh/', 'rf/', 'rm/', 'rh/')):
                d.qpos[:] = q
                d.qvel[:] = 0
                d.act[:] = 0
                d.ctrl[:] = 0
                mujoco.mj_forward(m, d)
                joint = m.joint(prefix + 'joint_LFTibia_pitch')
                dof = int(joint.dofadr[0])
                # Tarsus1 origin is the distal tibial joint, not the toe tip.
                tip_body = m.body(prefix + 'LFTarsus1').id
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, d.xpos[tip_body], tip_body)
                lever = float(np.linalg.norm(jp[:, dof]))
                assert lever > 1e-8
                baseline = float(d.qfrc_actuator[dof])
                actuator = m.actuator(prefix + 'LFTibia_flex_93434').id
                d.act[actuator] = 1
                mujoco.mj_forward(m, d)
                active_torque = float(d.qfrc_actuator[dof] - baseline)
                weight = float(m.body_mass.sum() * np.linalg.norm(m.opt.gravity))
                force = abs(active_torque) / lever
                results.append(dict(kind=kind, pose_index=pose_index, leg=prefix or 'source_left_foreleg',
                    tibia_q=float(d.qpos[int(joint.qposadr[0])]), lever_model_units=lever,
                    active_torque=active_torque, passive_actuator_torque=baseline,
                    tip_equivalent_active_force=force, body_weight_model_units=weight,
                    force_to_body_weight=force/weight))
    report = dict(kind=__doc__, runner_sha256=sha(Path(__file__)), mujoco=mujoco.__version__,
        inputs={name: sha(getattr(a, name)) for name in ('source_xml', 'xml', 'poses')},
        results=results,
        biological_reference=dict(url='https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/',
            citation='Azevedo et al. (2020), A size principle for recruitment of Drosophila leg motor neurons',
            observation='Approximately 100 micronewtons at the tibial tip, approximately ten body weights.',
            caveat='Observed voluntary peak across the experiment, not maximum isometric force at each modeled pose.'),
        assumptions=['Activation one of the single published tibia flexor unit; passive actuator torque subtracted.',
                     'Equivalent force tangential to tibia-tip rotation, holding all other coordinates fixed.',
                     'Ratio uses each model own total body weight and requires no SI mass conversion.',
                     'Muscle coverage, pose, probe direction and specimen differ from the experiment.',
                     'No parameter is fitted or changed, and no whole-body support or neural function is demonstrated.'])
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps([dict(kind=r['kind'],pose=r['pose_index'],leg=r['leg'],ratio=r['force_to_body_weight']) for r in results]))


if __name__ == '__main__':
    main()
