"""Uncalibrated antagonist muscle approximation, not a gait controller.

Named Tr/Ti flexor/extensor MN cohorts drive twelve hinges across all six legs.
No desired trajectory, oscillator, body orientation or task input is accepted.
Rest angle is fixed from the pinned rig pose, not adjusted during execution.
"""
import numpy as np


class AggregateLegMuscles:
    def __init__(self, annotation, motor_rows, scene, polarity=1):
        if polarity not in (-1,1):
            raise ValueError('polarity must be an explicit candidate sign')
        self.polarity = polarity
        index = {int(row):i for i,row in enumerate(motor_rows)}
        if len(index) != len(motor_rows):
            raise ValueError('duplicate motor rows')
        resident = scene['residents'][0]
        self.groups = []
        for side, prefix in (('L','l'),('R','r')):
            for subclass, leg in (('fl','f'),('ml','m'),('hl','h')):
                tag = prefix+leg
                for kind, joint in (('Tr',f'{tag}_coxa-{tag}_trochanterfemur-pitch'),
                                    ('Ti',f'{tag}_trochanterfemur-{tag}_tibia-pitch')):
                    channels = [i for i,a in enumerate(resident['actuators90'])
                                if a['semantic_id'] == joint+'-position']
                    if len(channels) != 1 or channels[0] >= 84:
                        raise ValueError('missing or ambiguous joint actuator')
                    cohorts = []
                    ids = []
                    for action in ('flexor','extensor'):
                        selected = [r for r in annotation['motors'] if r['subclass']==subclass and
                                    (r['somaSide'] or r['rootSide'])==side and
                                    r['mancType']==f'{kind} {action} MN']
                        if not selected:
                            raise ValueError('missing named motor cohort')
                        cohorts.append([index[r['cns_row']] for r in selected])
                        ids.append([r['bodyId'] for r in selected])
                    addresses = []
                    for body in scene['residents']:
                        matches = [j for j in body['joint_dofs126'] if j['semantic_id']==joint]
                        if len(matches)!=1:
                            raise ValueError('resident joint mismatch')
                        addresses.append((matches[0]['qpos_address'],matches[0]['dof_address']))
                    self.groups.append(dict(joint=joint, channel=channels[0], cohorts=cohorts,
                                            body_ids=ids, addresses=addresses,
                                            rest=resident['actuators90'][channels[0]]['neutral']))
        self.activation = np.zeros((len(scene['residents']),len(self.groups),2),dtype=np.float64)

    def step(self, rates, qpos, qvel, zero_motor=False):
        rates = np.asarray(rates)
        if rates.ndim!=2 or rates.shape[1]!=len(self.activation) or not np.isfinite(rates).all() or np.any((rates<0)|(rates>1)):
            raise ValueError('invalid normalized motor activity')
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError('invalid mechanical state')
        command = np.zeros((len(self.activation),92),dtype=np.float32)
        for lane in range(len(self.activation)):
            for muscle, group in enumerate(self.groups):
                target = np.zeros(2) if zero_motor else np.array([rates[c,lane].mean() for c in group['cohorts']])
                # Exact first-order recruitment over a 10-ms neural interval;
                # assumed 20-ms activation time constant, not fitted physiology.
                self.activation[lane,muscle] += (1-np.exp(-.01/.02))*(target-self.activation[lane,muscle])
                flexor, extensor = self.activation[lane,muscle]
                pos, vel = group['addresses'][lane]
                # Normalized by the host's original force capacity. The fixed
                # elastic rest pose is an assumption, not posture evidence.
                torque = (.03*self.polarity*(flexor-extensor)
                          + .03*(flexor+extensor)*(group['rest']-qpos[pos])
                          - .001*(flexor+extensor)*qvel[vel])
                command[lane,group['channel']] = np.clip(torque,-1,1)
        return command
