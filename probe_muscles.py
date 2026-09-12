"""Audit a published muscle component; not a whole-fly behavioral assay."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

# Anatomical name correspondences across specimens, not measured junction weights.
NAMED = {
    'LFC_pleural_remotor_and_abductor': 'Pleural remotor/abductor MN',
    'LFC_sternal_anterior_rotator': 'Sternal anterior rotator MN',
    'LFC_sternal_posterior_rotator': 'Sternal posterior rotator MN',
    'LFC_sternal_adductor': 'Sternal adductor MN',
    'LFF_accesory_trochanter_flexor': 'Acc. tr flexor MN',
    'LFF_trochanter_extensor': 'Tr extensor MN',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--neurons', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    annotation = json.loads(args.neurons.read_text())
    motors = [r for r in annotation['motors'] if r['subclass'] == 'fl' and
              (r['somaSide'] or r['rootSide']) == 'L']
    xml = ET.parse(args.xml)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    rows = []
    for index in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        if model.actuator_dyntype[index] != mujoco.mjtDyn.mjDYN_MUSCLE:
            raise ValueError(f'non-muscle actuator: {name}')
        selected = [r for r in motors if r['mancType'] == NAMED.get(name)]
        # Compare each muscle pulse to a zero-command baseline from exactly the
        # same keyframe. No trajectory tracking, joint target or learned policy.
        final = []
        for activation in (0., .05):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.ctrl[:] = 0
            data.ctrl[index] = activation
            peak_force = 0.
            for _ in range(100):
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.actuator_force).all():
                    raise RuntimeError(f'nonfinite muscle probe: {name}')
                peak_force = max(peak_force, abs(float(data.actuator_force[index])))
            final.append((data.qpos.copy(), peak_force))
        rows.append(dict(muscle=name, muscle_id=index,
            candidate_motor_rows=[r['cns_row'] for r in selected],
            candidate_body_ids=[r['bodyId'] for r in selected],
            annotation_label=NAMED.get(name),
            mapping_status='cross-specimen anatomical-name candidate; not calibrated' if selected else 'unresolved',
            pulse_minus_baseline_joint_radians=(final[1][0]-final[0][0]).tolist(),
            peak_force_model_units=final[1][1]))
    report = dict(
        kind='open-loop muscle component assay; NOT MaleCNS behavior',
        xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        neuron_source_sha256=annotation['source_sha256'], mujoco=mujoco.__version__,
        actuators=model.nu, joints=model.njnt, constraints=model.neq,
        free_joints=int(np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
        muscles=rows, explicit_assumptions=[
            'Fixed-base left-foreleg model, not a freely moving six-legged body.',
            'Named muscles are inter-specimen correspondences; no exact recruitment weights are asserted.',
            'Unresolved promotor subdivisions, flexor subdivisions, composite muscles and fast/slow motor units remain unassigned.',
            'Default muscle controls clamp below 0.0001; zero command is not exact zero activation.',
            '0.05 activation pulses for 0.01 seconds test mechanics, not endogenous behavior.',
        ])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(actuators=model.nu, free_joints=report['free_joints'],
        named_candidates=sum(bool(r['candidate_motor_rows']) for r in rows),
        responsive_muscles=sum(bool(np.max(np.abs(r['pulse_minus_baseline_joint_radians'])) > 1e-8) for r in rows)), indent=2))


if __name__ == '__main__':
    main()
