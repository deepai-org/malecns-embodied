import unittest
import xml.etree.ElementTree as ET
from make_passive_variant import transform


class TestPassiveVariant(unittest.TestCase):
    XML = b'<mujoco><worldbody><body><joint name="root" type="free"/><joint name="hinge" stiffness="10" damping="0.5" springref="0.3" axis="1 0 0"/></body></worldbody><actuator><general name="motor" gainprm="5"/></actuator></mujoco>'

    def test_only_requested_mechanics_change(self):
        raw,changes = transform(self.XML,0,1)
        before,after = ET.fromstring(self.XML),ET.fromstring(raw)
        for x,y in zip(before.iter(),after.iter()):
            self.assertEqual(x.tag,y.tag)
            expected = dict(x.attrib)
            if x.get('name')=='hinge':
                expected['stiffness']='0'
            self.assertEqual(expected,y.attrib)
        self.assertEqual(len(changes),1)
        self.assertEqual(changes[0]['after'],dict(stiffness=0,damping=.5))

    def test_invalid_scale_and_missing_explicit_values_rejected(self):
        for value in (-1,2,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                transform(self.XML,value,1)
        with self.assertRaises(ValueError):
            transform(self.XML.replace(b' stiffness="10"',b''),0,1)


if __name__ == '__main__':
    unittest.main()
