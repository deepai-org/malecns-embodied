import unittest
import numpy as np
try:
    import mujoco
    from scipy.optimize import linprog
    from probe_static_support import contact_rays
except ImportError:
    mujoco=None


@unittest.skipIf(mujoco is None,'requires MuJoCo and SciPy; run on experiment host')
class TestContactRays(unittest.TestCase):
    def test_box_gravity_and_force_direction(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/>
        <worldbody><geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        matrix=np.stack([r for c in data.contact for r in contact_rays(model,data,c)],axis=1)
        result=linprog(np.ones(matrix.shape[1]),A_eq=matrix,b_eq=data.qfrc_bias,bounds=(0,None),method='highs')
        self.assertTrue(result.success)
        np.testing.assert_allclose(matrix@result.x,data.qfrc_bias,atol=1e-8)
        wrong=linprog(np.ones(matrix.shape[1]),A_eq=-matrix,b_eq=data.qfrc_bias,bounds=(0,None),method='highs')
        self.assertEqual(wrong.status,2)


if __name__=='__main__':unittest.main()
