import unittest
import numpy as np
from run_muscle_feedback import position_current


class TestPositionCurrent(unittest.TestCase):
    def test_opponent_range_and_polarity(self):
        for angle in (-1, 0, .5, 1, 2):
            positive = position_current(angle, (0, 1), .2, 1)
            negative = position_current(angle, (0, 1), .2, -1)
            np.testing.assert_array_equal(positive, negative[::-1])
            self.assertAlmostEqual(float(positive.sum()), .2)
            self.assertTrue(np.all((positive >= 0) & (positive <= .2)))
        np.testing.assert_array_equal(position_current(0, (0, 1), .2, 1), [0, .2])
        np.testing.assert_array_equal(position_current(1, (0, 1), .2, 1), [.2, 0])

    def test_invalid_inputs_rejected(self):
        for args in ((np.nan, (0, 1), .2, 1), (0, (1, 1), .2, 1),
                     (0, (0, 1), -1, 1), (0, (0, 1), .2, 0)):
            with self.assertRaises(ValueError):
                position_current(*args)


if __name__ == '__main__':
    unittest.main()
