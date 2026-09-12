import copy
import unittest
import numpy as np
from named_leg_senses import NamedLegPosition


class TestNamedSenses(unittest.TestCase):
    def fixture(self):
        joints, actuators, neurons = [], [], []
        for i, (side, tag, nerve) in enumerate([('L','lf','ProLN'),('L','lm','MesoLN'),('L','lh','MetaLN'),('R','rf','ProLN'),('R','rm','MesoLN'),('R','rh','MetaLN')]):
            name = f'{tag}_trochanterfemur-{tag}_tibia-pitch'
            joints.append(dict(semantic_id=name,qpos_address=i))
            actuators.append(dict(semantic_id=name+'-position',neutral=0.))
            for k, kind in enumerate(('SNpp50','SNpp51')):
                neurons.append(dict(type=kind,entryNerve=nerve,somaSide=side,rootSide=None,cns_row=2*i+k,bodyId=100+2*i+k))
        return dict(neurons=neurons), dict(residents=[dict(joint_dofs126=joints,actuators90=actuators)])

    def test_local_opponent_and_polarity(self):
        a,s = self.fixture()
        sensor = NamedLegPosition(a,s,range(12),[99])
        np.testing.assert_array_equal(sensor.offsets(np.zeros(6)),0)
        q = np.zeros(6); q[0]=1
        values = sensor.offsets(q)
        self.assertGreater(values[0,0],0)
        self.assertEqual(values[0,0],-values[1,0])
        np.testing.assert_array_equal(values[2:],0)
        np.testing.assert_array_equal(NamedLegPosition(a,s,range(12),[99],-1).offsets(q),-values)
        self.assertLessEqual(np.abs(sensor.offsets(np.ones(6)*100)).max(),.200001)

    def test_missing_and_invalid_rows(self):
        a,s = self.fixture(); a['neurons'].pop()
        sensor = NamedLegPosition(a,s,range(12),[99])
        self.assertEqual(sensor.missing,[dict(leg='rh',type='SNpp51')])
        with self.assertRaises(ValueError):
            NamedLegPosition(a,s,range(12),[0])
        bad=copy.deepcopy(a);bad['neurons'][1]['cns_row']=0
        with self.assertRaises(ValueError):
            NamedLegPosition(bad,s,range(12),[99])
        with self.assertRaises(ValueError):
            sensor.offsets(np.full(6,np.nan))


if __name__ == '__main__':
    unittest.main()
