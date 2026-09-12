"""Extend aggregate legs with named anterior/posterior coxa rotators.

Fixed one-axis mechanics per coxa; no gait or target-position controller.
"""
import numpy as np
from aggregate_leg_muscles import AggregateLegMuscles


class CoxaLegMuscles(AggregateLegMuscles):
    def __init__(self, annotation, motor_rows, scene, mapping, polarity=1):
        super().__init__(annotation,motor_rows,scene,polarity)
        index={int(row):i for i,row in enumerate(motor_rows)}
        if {g['leg'] for g in mapping['groups']} != {'lf','lm','lh','rf','rm','rh'} or len(mapping['groups'])!=6:
            raise ValueError('expected six coxa coordinate mappings')
        for m in mapping['groups']:
            tag=m['leg']; side=tag[0].upper(); subclass={'f':'fl','m':'ml','h':'hl'}[tag[1]]
            if m['positive_protraction_sign'] not in (-1,1):
                raise ValueError('invalid fixed coordinate sign')
            joint=m['joint']
            if joint not in [f'c_thorax-{tag}_coxa-{axis}' for axis in ('yaw','pitch','roll')]:
                raise ValueError('invalid coxa target')
            cohorts=[];ids=[]
            labels=['Sternal anterior rotator MN','Sternal posterior rotator MN']
            if m['positive_protraction_sign']<0:labels.reverse()
            for label in labels:
                selected=[r for r in annotation['motors'] if r['mancType']==label and r['subclass']==subclass and (r['somaSide'] or r['rootSide'])==side]
                if not selected or any(r['cns_row'] not in index for r in selected):
                    raise ValueError('missing named coxa motor cohort')
                cohorts.append([index[r['cns_row']] for r in selected]);ids.append([r['bodyId'] for r in selected])
            channels=[i for i,x in enumerate(scene['residents'][0]['actuators90']) if x['semantic_id']==joint+'-position']
            if len(channels)!=1 or channels[0]>=84 or channels[0] in {g['channel'] for g in self.groups}:
                raise ValueError('invalid or duplicate actuator')
            addresses=[]
            for resident in scene['residents']:
                matches=[j for j in resident['joint_dofs126'] if j['semantic_id']==joint]
                if len(matches)!=1:raise ValueError('missing physical joint')
                addresses.append((matches[0]['qpos_address'],matches[0]['dof_address']))
            self.groups.append(dict(joint=joint,channel=channels[0],cohorts=cohorts,body_ids=ids,
                addresses=addresses,rest=scene['residents'][0]['actuators90'][channels[0]]['neutral'],
                cohort_labels=labels,coordinate_sign=m['positive_protraction_sign']))
        self.activation=np.zeros((len(scene['residents']),len(self.groups),2),dtype=np.float64)
