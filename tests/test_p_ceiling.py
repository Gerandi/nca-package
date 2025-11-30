import math
import os
import sys
import unittest

# Add the parent directory to sys.path to import nca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nca.p_ceiling import p_ceiling


class TestPCeiling(unittest.TestCase):
    def setUp(self):
        self.loop_data = {
            "scope_theo": [1, 6, 2, 6],
            "scope_area": 20,
            "flip_x": False,
            "flip_y": False,
        }

    def assertEqualFloat(self, first, second, places=7):
        if math.isnan(first) and math.isnan(second):
            return
        self.assertAlmostEqual(first, second, places=places)

    def test_upper_left_corner(self):
        # Upper left upper corner
        self.loop_data["flip_x"] = False
        self.loop_data["flip_y"] = False

        self.assertEqualFloat(p_ceiling(self.loop_data, 1, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, -4 / 3, 8), float("nan"))

        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 4), 4 / 3)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 3), 15)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 5), 5)
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 4), float("nan"))

        self.assertEqualFloat(p_ceiling(self.loop_data, 4 / 3, 0), 8)
        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 0), 17)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 0), 20)
        self.assertEqualFloat(p_ceiling(self.loop_data, -1, 0), float("nan"))

    def test_upper_right_corner(self):
        # Upper right corner
        self.loop_data["flip_x"] = True
        self.loop_data["flip_y"] = False

        self.assertEqualFloat(p_ceiling(self.loop_data, 1, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 8), 3)
        self.assertEqualFloat(p_ceiling(self.loop_data, -4 / 3, 8), 12)

        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 4), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 3), 15)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 5), 5)
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 4), 56 / 3)

        self.assertEqualFloat(p_ceiling(self.loop_data, 4 / 3, 0), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 0), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 0), 20)
        self.assertEqualFloat(p_ceiling(self.loop_data, -1, 0), 20)

    def test_lower_right_corner(self):
        # Lower right corner
        self.loop_data["flip_x"] = True
        self.loop_data["flip_y"] = True

        self.assertEqualFloat(p_ceiling(self.loop_data, 1, 8), 20)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 8), 20)
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, -4 / 3, 8), float("nan"))

        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 4), 56 / 3)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 3), 5)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 5), 15)
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 4), float("nan"))

        self.assertEqualFloat(p_ceiling(self.loop_data, 4 / 3, 0), 12)
        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 0), 3)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 0), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, -1, 0), float("nan"))

    def test_lower_left_corner(self):
        # Lower left corner
        self.loop_data["flip_x"] = False
        self.loop_data["flip_y"] = True

        self.assertEqualFloat(p_ceiling(self.loop_data, 1, 8), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 8), 20)
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 8), 17)
        self.assertEqualFloat(p_ceiling(self.loop_data, -4 / 3, 8), 8)

        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 4), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 3), 5)
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 5), 15)
        self.assertEqualFloat(p_ceiling(self.loop_data, -2 / 3, 4), 4 / 3)

        self.assertEqualFloat(p_ceiling(self.loop_data, 4 / 3, 0), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 2 / 3, 0), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, 0, 0), float("nan"))
        self.assertEqualFloat(p_ceiling(self.loop_data, -1, 0), float("nan"))


if __name__ == "__main__":
    unittest.main()
