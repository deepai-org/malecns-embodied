import unittest
import numpy as np
from aggregate_leg_muscles import AggregateLegMuscles


def fixture():
    motors, actuators, joints = [], [], []
    for side,prefix in (('L','l'),('R','r')):
        for subclass,leg in (('fl','f'),('ml','m'),('hl','h')):
            tag = prefix+leg
            for kind,joint in (('Tr',f'{tag}_coxa-{tag}_trochanterfemur-pitch'),
                               ('Ti',f'{tag}_trochanterfemur-{tag}_tibia-pitch')):
                address = len(joints)
                joints.append(dict(semantic_id=joint,qpos_address=address,dof_address=address))
                actuators.append(dict(semantic_id=joint+'-position',neutral=0))
                for action in ('flexor','extensor'):
                    row = len(motors)
                    motors.append(dict(subclass=subclass,somaSide=side,rootSide=side,
                                       mancType=f'{kind} {action} MN',cns_row=row,bodyId=row+100))
    return dict(motors=motors),np.arange(len(motors)),dict(residents=[dict(actuators90=actuators,joint_dofs126=joints)])


class TestAggregateMuscles(unittest.TestCase):
    def test_zero_neural_input_removes_active_force(self):
        d = AggregateLegMuscles(*fixture())
        output = d.step(np.zeros((24,1)),np.ones(12),np.ones(12))
        np.testing.assert_array_equal(output,0)

    def test_named_cohort_isolation_and_polarity(self):
        rates = np.zeros((24,1)); rates[0] = .5
        d = AggregateLegMuscles(*fixture())
        out = d.step(rates,np.zeros(12),np.zeros(12))
        self.assertGreater(out[0,0],0)
        np.testing.assert_array_equal(out[0,1:],0)
        reverse = AggregateLegMuscles(*fixture(),polarity=-1)
        np.testing.assert_array_equal(reverse.step(rates,np.zeros(12),np.zeros(12)),-out)

    def test_cocontraction_fixed_elastic_rest(self):
        d = AggregateLegMuscles(*fixture())
        out = d.step(np.full((24,1),.2),np.full(12,.1),np.zeros(12))
        self.assertTrue(np.all(out[0,:12]<0))
        np.testing.assert_array_equal(out[0,12:],0)

    def test_missing_cohort_and_invalid_rate_rejected(self):
        a,rows,scene = fixture(); a['motors'].pop()
        with self.assertRaises(ValueError):
            AggregateLegMuscles(a,rows,scene)
        d = AggregateLegMuscles(*fixture())
        with self.assertRaises(ValueError):
            d.step(np.full((24,1),np.nan),np.zeros(12),np.zeros(12))


if __name__ == '__main__':
    unittest.main()
