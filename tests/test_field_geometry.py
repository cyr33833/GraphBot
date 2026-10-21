import unittest

from core.field_geometry import (
    game_to_pixel,
    pixel_radius_to_game,
    pixel_scales,
    pixel_to_game,
    get_y_bounds,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    Y_HALF_DEFAULT,
)


class FieldGeometryTests(unittest.TestCase):
    def test_pixel_to_game_isotropic(self):
        # Center (385, 225) maps to (0.0, 0.0)
        self.assertEqual(pixel_to_game(385, 225, 770, 450), (0.0, 0.0))
        # Top-left (0, 0) maps to (-25.0, Y_HALF_DEFAULT) where Y_HALF_DEFAULT ~ 14.6104
        gx, gy = pixel_to_game(0, 0, 770, 450)
        self.assertAlmostEqual(gx, -25.0)
        self.assertAlmostEqual(gy, Y_HALF_DEFAULT, places=4)
        # Bottom-right (770, 450) maps to (25.0, -Y_HALF_DEFAULT)
        gx, gy = pixel_to_game(770, 450, 770, 450)
        self.assertAlmostEqual(gx, 25.0)
        self.assertAlmostEqual(gy, -Y_HALF_DEFAULT, places=4)

    def test_game_to_pixel_is_inverse(self):
        self.assertEqual(game_to_pixel(0, 0, 770, 450), (385.0, 225.0))
        px, py = game_to_pixel(-25, Y_HALF_DEFAULT, 770, 450)
        self.assertAlmostEqual(px, 0.0)
        self.assertAlmostEqual(py, 0.0)
        px, py = game_to_pixel(25, -Y_HALF_DEFAULT, 770, 450)
        self.assertAlmostEqual(px, 770.0)
        self.assertAlmostEqual(py, 450.0)

    def test_radius_and_scales_isotropic(self):
        scale_x, scale_y = pixel_scales(770, 450)
        self.assertEqual(scale_x, scale_y)
        self.assertAlmostEqual(scale_x, 50.0 / 770.0)
        radius = pixel_radius_to_game(10, 770, 450)
        expected = 10 * 50.0 / 770.0
        self.assertAlmostEqual(radius, expected)


if __name__ == "__main__":
    unittest.main()
