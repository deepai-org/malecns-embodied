"""Named claw afferent position transduction; explicit uncalibrated approximation."""
import numpy as np


class NamedLegPosition:
    def __init__(self, annotation, scene, body_rows, motor_rows, polarity=1, amplitude=.2):
        if polarity not in (-1, 1) or not 0 < amplitude <= .5:
            raise ValueError('invalid sensory parameters')
        self.polarity, self.amplitude = polarity, amplitude
        self.groups, self.missing = [], []
        allowed, forbidden = set(map(int, body_rows)), set(map(int, motor_rows))
        used = set()
        for side, prefix in [('L', 'l'), ('R', 'r')]:
            for leg, nerve in [('f', 'ProLN'), ('m', 'MesoLN'), ('h', 'MetaLN')]:
                tag = prefix+leg
                joint = f'{tag}_trochanterfemur-{tag}_tibia-pitch'
                addresses = []
                for resident in scene['residents']:
                    matches = [j for j in resident['joint_dofs126'] if j['semantic_id'] == joint]
                    if len(matches) != 1:
                        raise ValueError('missing or ambiguous physical joint')
                    addresses.append(matches[0]['qpos_address'])
                actuators = [x for x in scene['residents'][0]['actuators90'] if x['semantic_id'] == joint+'-position']
                if len(actuators) != 1:
                    raise ValueError('missing rig rest reference')
                for kind, sign in [('SNpp50', 1), ('SNpp51', -1)]:
                    records = [r for r in annotation['neurons'] if r['type'] == kind and
                               r['entryNerve'] == nerve and (r['somaSide'] or r['rootSide']) == side]
                    if not records:
                        self.missing.append(dict(leg=tag, type=kind))
                        continue
                    rows = [r['cns_row'] for r in records]
                    if len(set(rows)) != len(rows) or any(r not in allowed or r in forbidden or r in used for r in rows):
                        raise ValueError('invalid or repeated sensory row')
                    used.update(rows)
                    self.groups.append(dict(leg=tag, type=kind, sign=sign, rows=rows,
                                            body_ids=[r['bodyId'] for r in records],
                                            qpos_addresses=addresses, rest=actuators[0]['neutral']))
        if not self.groups:
            raise ValueError('no named sensory groups')

    def offsets(self, qpos):
        qpos = np.asarray(qpos)
        if qpos.ndim != 1 or not np.isfinite(qpos).all():
            raise ValueError('invalid physical joint state')
        # 1-radian tuning scale and rig-neutral center are assumptions. No
        # time, target trajectory, motor output or task state enters this map.
        return np.asarray([self.amplitude*self.polarity*g['sign']*
                           np.tanh(qpos[g['qpos_addresses']]-g['rest'])
                           for g in self.groups], dtype=np.float32)
