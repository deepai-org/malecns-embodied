import unittest
import numpy as np
from aggregate_leg_muscles import AggregateLegMuscles
from test_aggregate_leg_muscles import fixture
from physics_muscles import muscle_coefficients


class TestPhysicsMuscles(unittest.TestCase):
    def test_matches_existing_law_at_same_state(self):
        for polarity in (-1,1):
            drive = AggregateLegMuscles(*fixture(),polarity=polarity)
            q=np.linspace(-1,1,12);v=np.linspace(-3,3,12)
            command=drive.step(np.linspace(0,1,24)[:,None],q,v)
            c=muscle_coefficients(drive)
            observed=np.clip(c[0,:12,0]-c[0,:12,1]*q-c[0,:12,2]*v,-1,1)
            np.testing.assert_allclose(observed,command[0,:12],rtol=0,atol=2e-9)
            np.testing.assert_array_equal(c[:,12:],0)

    def test_zero_and_invalid_activation(self):
        d=AggregateLegMuscles(*fixture())
        np.testing.assert_array_equal(muscle_coefficients(d),0)
        d.activation[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            muscle_coefficients(d)


if __name__=='__main__':
    unittest.main()
