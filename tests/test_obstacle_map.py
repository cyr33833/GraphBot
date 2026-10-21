import unittest
import numpy as np
import cv2

from core.avoidance import ObstacleMap, extract_obstacle_map, field_obstacles_to_game
from core.pathfinding import astar_game, build_enemy_chain_astar
from core.polynomial_planner import search_best_polynomial
from core.symbolic_ga_planner import search_best_symbolic_ga
from core.detection import detect_all_players_robust


class TestObstacleMap(unittest.TestCase):
    def setUp(self):
        # 800 x 600 canvas (game coords [-25, 25] x [-15, 15])
        # Center is (400, 300) -> (0, 0) in game coords
        self.w = 800
        self.h = 600
        self.img = np.full((self.h, self.w, 3), 255, dtype=np.uint8)
        # Draw a solid black circle at center with radius 40px
        cv2.circle(self.img, (400, 300), 40, (0, 0, 0), -1)
        self.obs_map = extract_obstacle_map(self.img)

    def test_obstacle_map_properties(self):
        self.assertEqual(self.obs_map.w, self.w)
        self.assertEqual(self.obs_map.h, self.h)
        self.assertGreaterEqual(len(self.obs_map.contours), 1)
        self.assertGreaterEqual(len(self.obs_map.boundary_circles), 10)

    def test_clearance_at_game(self):
        # Center (0, 0) is inside the black circle -> clearance should be 0.0
        c_center = self.obs_map.clearance_at_game(0.0, 0.0)
        self.assertEqual(c_center, 0.0)
        self.assertFalse(self.obs_map.is_point_safe(0.0, 0.0, clearance=0.1))

        # Far point (-20, 10) is free space -> clearance > 5.0
        c_far = self.obs_map.clearance_at_game(-20.0, 10.0)
        self.assertGreater(c_far, 5.0)
        self.assertTrue(self.obs_map.is_point_safe(-20.0, 10.0, clearance=0.25))

    def test_segment_collision(self):
        # Segment crossing the obstacle from (-10, 0) to (10, 0)
        self.assertFalse(self.obs_map.is_segment_safe((-10.0, 0.0), (10.0, 0.0), clearance=0.25))
        # Segment above the obstacle from (-10, 8) to (10, 8)
        self.assertTrue(self.obs_map.is_segment_safe((-10.0, 8.0), (10.0, 8.0), clearance=0.25))

    def test_backward_compatibility_iteration(self):
        # ObstacleMap can be iterated over as (gx, gy, r) tuples
        count = 0
        for gx, gy, r in self.obs_map:
            self.assertIsInstance(gx, float)
            self.assertIsInstance(gy, float)
            self.assertIsInstance(r, float)
            count += 1
            if count >= 5:
                break
        self.assertEqual(count, 5)

    def test_field_obstacles_to_game_identity(self):
        # field_obstacles_to_game returns the ObstacleMap itself if passed one
        res = field_obstacles_to_game(self.obs_map, self.w, self.h)
        self.assertIs(res, self.obs_map)

    def test_astar_with_obstacle_map(self):
        start = (-15.0, 0.0)
        goal = (15.0, 0.0)
        path = astar_game(start, goal, self.obs_map, clearance=0.35)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)
        # Ensure path does not go through the obstacle center
        for pt in path:
            self.assertTrue(self.obs_map.is_point_safe(pt[0], pt[1], clearance=0.0))

    def test_planners_run_without_error(self):
        enemies = [(12.0, 0.0, 0.6)]
        poly_res = search_best_polynomial(-15.0, 0.0, enemies, self.obs_map, [], budget_ms=40)
        self.assertIn("formula", poly_res)

        sym_res = search_best_symbolic_ga(-15.0, 0.0, enemies, self.obs_map, [], budget_ms=40)
        self.assertIn("formula", sym_res)


if __name__ == "__main__":
    unittest.main()
