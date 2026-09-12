"""Reconstruct geometric body/terrain contacts from recorded physical poses.

This does not reconstruct contact forces. It uses the pinned scene geometry;
it is not a substitute for native contact telemetry if geometry changes online.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    import mujoco
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    scene = json.loads(a.scene.read_text())
    receipt = json.loads((a.trial/'result.json').read_text())
    if not receipt['completed'] or receipt['scene_sha256']!=hashlib.sha256(a.scene.read_bytes()).hexdigest():
        raise ValueError('scene/trial identity mismatch')
    model = mujoco.MjModel.from_xml_path(str(a.scene.parent/scene['scene_xml']))
    data = mujoco.MjData(model)
    residents = [r['id'] for r in scene['residents']]
    first = {name:None for name in residents}
    terminal = {name:set() for name in residents}
    counts = {name:0 for name in residents}
    with np.load(a.trial/'trace.npz',allow_pickle=False) as trace:
        if trace['qpos'].shape[1]!=model.nq or trace['qvel'].shape[1]!=model.nv:
            raise ValueError('recorded geometry topology differs')
        times = trace['time']
        last_second = times >= times[-1]-1
        for tick,t in enumerate(times):
            data.qpos[:] = trace['qpos'][tick]
            data.qvel[:] = trace['qvel'][tick]
            mujoco.mj_forward(model,data)
            touching = {name:set() for name in residents}
            for contact in data.contact:
                if contact.dist > 0:
                    continue
                names = [model.geom(int(g)).name for g in contact.geom]
                for body,terrain in (names,names[::-1]):
                    if not terrain.startswith('ecology/'):
                        continue
                    for resident in residents:
                        if body.startswith(resident+'/c_thorax') or body.startswith(resident+'/c_abdomen'):
                            touching[resident].add((body,terrain))
            for resident in residents:
                if touching[resident] and first[resident] is None:
                    first[resident] = float(t)
                if last_second[tick] and touching[resident]:
                    counts[resident] += 1
                if tick==len(times)-1:
                    terminal[resident] = touching[resident]
        denominator = int(last_second.sum())
        report = dict(kind='geometric thorax/abdomen contact reconstruction; not contact-force measurement',
                      scene_sha256=receipt['scene_sha256'],
                      trace_sha256=hashlib.sha256((a.trial/'trace.npz').read_bytes()).hexdigest(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      first_core_terrain_contact_seconds=first,
                      last_second_core_contact_fraction={k:v/denominator for k,v in counts.items()},
                      final_core_contacts={k:sorted(v) for k,v in terminal.items()},
                      initial_thorax_height_mm=trace['thorax_position'][0,:,2].tolist(),
                      final_thorax_height_mm=trace['thorax_position'][-1,:,2].tolist(),
                      caveat='Original scene meshes are used; dynamic ecology geometry changes are not reconstructed.')
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
