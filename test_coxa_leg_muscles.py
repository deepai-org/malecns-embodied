import unittest
import numpy as np
from test_aggregate_leg_muscles import fixture
from coxa_leg_muscles import CoxaLegMuscles


def coxa_fixture():
    a,_,s=fixture();groups=[];resident=s['residents'][0]
    for side,prefix in [('L','l'),('R','r')]:
        for subclass,leg in [('fl','f'),('ml','m'),('hl','h')]:
            tag=prefix+leg;joint=f'c_thorax-{tag}_coxa-roll';address=len(resident['joint_dofs126'])
            resident['joint_dofs126'].append(dict(semantic_id=joint,qpos_address=address,dof_address=address))
            resident['actuators90'].append(dict(semantic_id=joint+'-position',neutral=0.))
            groups.append(dict(leg=tag,joint=joint,positive_protraction_sign=-1))
            for label in ('Sternal anterior rotator MN','Sternal posterior rotator MN'):
                row=len(a['motors']);a['motors'].append(dict(subclass=subclass,somaSide=side,rootSide=None,mancType=label,cns_row=row,bodyId=100+row))
    return a,np.arange(len(a['motors'])),s,dict(groups=groups)


class TestCoxa(unittest.TestCase):
    def test_named_drive_sign_and_isolation(self):
        d=CoxaLegMuscles(*coxa_fixture());rates=np.zeros((36,1));rates[24]=.5
        out=d.step(rates,np.zeros(18),np.zeros(18))
        self.assertLess(out[0,12],0)
        np.testing.assert_array_equal(np.delete(out[0],12),0)
        self.assertEqual(len(d.groups),18)

    def test_zero_and_mapping_rejection(self):
        d=CoxaLegMuscles(*coxa_fixture())
        np.testing.assert_array_equal(d.step(np.zeros((36,1)),np.ones(18),np.ones(18)),0)
        a,r,s,m=coxa_fixture();m['groups'][0]['joint']='c_thorax-rf_coxa-roll'
        with self.assertRaises(ValueError):CoxaLegMuscles(a,r,s,m)
        a,r,s,m=coxa_fixture();a['motors'].pop()
        with self.assertRaises(ValueError):CoxaLegMuscles(a,r,s,m)


if __name__=='__main__':unittest.main()
