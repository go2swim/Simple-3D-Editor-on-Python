import numpy as np
import unittest

from src.intersections import (
    find_intersection_2d,
    point_on_line,
    LocalSystemCoord,
    get_intersection_line_and_point_of_two_planes,
)
from src.premitives import Plane


class TestFindIntersection(unittest.TestCase):

    def test_simple_intersection(self):
        line1 = np.array([[0, 0], [4, 4]])
        line2 = np.array([[0, 4], [4, 0]])
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        expected = np.array([2, 2])
        self.assertTrue(
            np.allclose(intersection, expected),
            f"Expected {expected}, but got {intersection}",
        )

    def test_parallel_lines(self):
        line1 = np.array([[0, 0], [4, 0]])
        line2 = np.array([[0, 1], [4, 1]])
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        self.assertIsNone(intersection, f"Expected None, but got {intersection}")

    def test_intersection_outside_segments(self):
        line1 = np.array([[0, 0], [2, 2]])
        line2 = np.array([[3, 3], [5, 5]])
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        self.assertIsNone(intersection, f"Expected None, but got {intersection}")

    def test_intersection_on_border(self):
        line1 = np.array([[0, 0], [4, 4]])
        line2 = np.array([[4, 4], [6, 2]])
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        expected = np.array([4, 4])
        self.assertTrue(
            np.allclose(intersection, expected),
            f"Expected {expected}, but got {intersection}",
        )

    def test_perpendicular_intersection(self):
        line1 = np.array([[0, 0], [0, 4]])
        line2 = np.array([[-2, 2], [2, 2]])
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        expected = np.array([0, 2])
        self.assertTrue(
            np.allclose(intersection, expected),
            f"Expected {expected}, but got {intersection}",
        )


class TestPointOnLine2D(unittest.TestCase):

    def test_point_on_line_inside_segment(self):
        line = np.array([[0, 0], [2, 2]])
        point = np.array([1, 1])
        self.assertTrue(point_on_line(point, line))

    def test_point_on_line_at_endpoint(self):
        line = np.array([[0, 0], [2, 2]])
        point = np.array([2, 2])
        self.assertTrue(point_on_line(point, line))

    def test_point_on_line_outside_segment(self):
        line = np.array([[0, 0], [2, 2]])
        point = np.array([3, 3])
        self.assertFalse(point_on_line(point, line))

    def test_point_not_on_line(self):
        line = np.array([[0, 0], [2, 2]])
        point = np.array([1, 2])
        self.assertFalse(point_on_line(point, line))

    def test_line_parallel_to_axis(self):
        line = np.array([[0, 0], [0, 2]])
        point = np.array([0, 1])
        self.assertTrue(point_on_line(point, line))

    def test_line_with_zero_direction(self):
        line = np.array([[0, 0], [0, 0]])
        point = np.array([0, 0])
        self.assertTrue(point_on_line(point, line))
        point = np.array([1, 0])
        self.assertFalse(point_on_line(point, line))


class TestLocalSystemCoord(unittest.TestCase):

    def test_to_local_coord(self):
        edge = np.array([1, 0, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([1, 1, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        point = np.array([2, 2, 0])
        local_point = local_system.to_local_coord(point)
        expected_local_point = np.array([1, 1])

        self.assertTrue(
            np.allclose(local_point, expected_local_point),
            f"Expected local coordinates {expected_local_point}, but got {local_point}",
        )

    def test_to_global_coord(self):
        edge = np.array([1, 0, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([1, 1, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        local_point = np.array([1, 1])
        global_point = local_system.to_global_coord(local_point)
        expected_global_point = np.array([2, 2, 0])

        self.assertTrue(
            np.allclose(global_point, expected_global_point),
            f"Expected global coordinates {expected_global_point}, but got {global_point}",
        )

    def test_round_trip_conversion(self):
        edge = np.array([2, 0, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([3, 3, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        point = np.array([5, 7, 0])
        local_point = local_system.to_local_coord(point)
        global_point = local_system.to_global_coord(local_point)

        self.assertTrue(
            np.allclose(global_point, point),
            f"Expected global coordinates {point}, but got {global_point}",
        )

    def test_non_orthogonal_axes(self):
        edge = np.array([1, 1, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([0, 0, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        point = np.array([2, 2, 0])
        local_point = local_system.to_local_coord(point)
        global_point = local_system.to_global_coord(local_point)

        self.assertTrue(
            np.allclose(global_point, point),
            f"Expected global coordinates {point}, but got {global_point}",
        )


class TestFindPointAndLineIntersection(unittest.TestCase):

    def test_simple_intersection(self):

        plane1 = Plane()
        plane1.corners = np.array([[1, 0, 1], [1, 0, 0], [-1, 0, 1], [-1, 0, 0]])

        plane2 = Plane()
        plane2.corners = np.array([[0, 1, 1], [0, 1, 0], [0, -1, 1], [0, -1, 0]])

        expected_dir = np.array([0, 0, 1])

        intersection_point, line_direction = (
            get_intersection_line_and_point_of_two_planes(plane1, plane2)
        )

        line_direction = line_direction / np.linalg.norm(line_direction)
        expected_dir = expected_dir / np.linalg.norm(expected_dir)

        self.assertTrue(
            np.allclose(line_direction, expected_dir)
            or np.allclose(line_direction, -expected_dir)
        )

        self.assertTrue(
            np.allclose(np.dot(intersection_point, line_direction), 0, atol=1e-6)
        )

    def test_intersection_with_translation(self):

        plane1 = Plane()
        plane1.corners = np.array([[1, 0, 1], [1, 0, 0], [-1, 0, 1], [-1, 0, 0]])

        plane2 = Plane()
        plane2.corners = np.array([[0, 1, 1], [0, 1, 0], [0, -1, 1], [0, -1, 0]])

        plane1.translate(0, 0, 2)

        expected_dir = np.array([0, 0, 1])

        intersection_point, line_direction = (
            get_intersection_line_and_point_of_two_planes(plane1, plane2)
        )

        line_direction = line_direction / np.linalg.norm(line_direction)
        expected_dir = expected_dir / np.linalg.norm(expected_dir)

        self.assertTrue(
            np.allclose(line_direction, expected_dir)
            or np.allclose(line_direction, -expected_dir)
        )

        self.assertTrue(
            np.allclose(np.dot(intersection_point, line_direction), 0, atol=1e-6)
        )

    def test_non_intersecting_planes(self):

        plane1 = Plane()
        plane1.corners = np.array([[1, 0, 1], [1, 0, 0], [-1, 0, 1], [-1, 0, 0]])

        plane2 = Plane()
        plane2.corners = np.array([[1, 1, 1], [1, 1, 0], [-1, 1, 1], [-1, 1, 0]])

        result = get_intersection_line_and_point_of_two_planes(plane1, plane2)
        self.assertIsNone(result[0])


if __name__ == "__main__":
    unittest.main()
