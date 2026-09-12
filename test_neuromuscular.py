import unittest
import numpy as np
from neuromuscular import NamedMuscleDrive


class TestNamedMuscleDrive(unittest.TestCase):
    def setUp(self):
        self.report = dict(actuators=3, muscles=[
            dict(muscle_id=0, muscle='a', candidate_motor_rows=[10, 30]),
            dict(muscle_id=1, muscle='b', candidate_motor_rows=[20]),
            dict(muscle_id=2, muscle='unresolved', candidate_motor_rows=[]),
        ])
        self.drive = NamedMuscleDrive(self.report, [10, 20, 30, 40])

    def test_zero_activity_zero_neural_drive(self):
        np.testing.assert_array_equal(self.drive(np.zeros(4)), np.zeros(3))

    def test_only_named_neurons_affect_each_muscle(self):
        np.testing.assert_allclose(self.drive([.2, .7, .4, 1]), [.3, .7, 0])
        np.testing.assert_allclose(self.drive([.2, .7, .4, 0]), [.3, .7, 0])

    def test_nonmotor_mapping_rejected(self):
        self.report['muscles'][0]['candidate_motor_rows'] = [999]
        with self.assertRaises(KeyError):
            NamedMuscleDrive(self.report, [10, 20, 30])

    def test_invalid_rates_rejected(self):
        for rates in ([0]*3, [np.nan]*4, [-.1]*4, [1.1]*4):
            with self.assertRaises(ValueError):
                self.drive(rates)


if __name__ == '__main__':
    unittest.main()
