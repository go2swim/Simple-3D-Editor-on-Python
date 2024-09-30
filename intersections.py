import numpy as np
import unittest

from node import get_point_coord
from premitives import Plane


def find_intersection_2d(line1, line2_direction, line2_intersection_point):
    local_start = line1[0]
    local_end = line1[1]
    edge_direction = local_end - local_start

    local_line_direction = line2_direction
    local_intersection_point = line2_intersection_point

    # Решаем уравнение для нахождения t в отрезке и s для линии пересечения
    A_local = np.array([local_line_direction, -edge_direction]).T
    b_local = local_start - local_intersection_point

    try:
        t_s_solution = np.linalg.solve(A_local, b_local)
        t = t_s_solution[0]
        s = t_s_solution[1]
    except np.linalg.LinAlgError:
        return None

    return local_start + edge_direction * s

def point_on_line(point, line):
    origin = line[0]
    direction = line[1] - line[0]
    coefc = []
    for i in range(2):
        if direction[i] != 0:
            coef = (point[i] - origin[i]) / direction[i]
            coefc.append(coef)
            if not (0 <= coef <= 1):
                return False
        else:
            if point[i] != origin[i]:
                return False

    return 0 <= len(coefc) <= 1 or coefc[0] == coefc[1]



class LocalSystemCoord:
    def __init__(self, edge, normal, origin):
        self.x_axis = edge / np.linalg.norm(edge)
        self.y_axis = np.cross(normal, self.x_axis) / np.linalg.norm(np.cross(normal, self.x_axis))
        self.origin = origin


    def to_local_coord(self, point):
        relative_point = point - self.origin
        x_coord = np.dot(relative_point, self.x_axis)
        y_coord = np.dot(relative_point, self.y_axis)
        return np.array([x_coord, y_coord])

    def to_global_coord(self, point):
        return self.origin + self.x_axis * point[0] + self.y_axis * point[1]


def get_intersection_line_and_point_of_two_planes(plane1, plane2):
    normal1 = np.cross(get_point_coord(plane1.corners[1] - plane1.corners[0], plane1),
                       get_point_coord(plane1.corners[2] - plane1.corners[0], plane1))
    normal2 = np.cross(get_point_coord(plane2.corners[1] - plane2.corners[0], plane2),
                       get_point_coord(plane2.corners[2] - plane2.corners[0], plane2))

    # находим направление линии пересечения (векторное произведение нормалей)
    line_direction = np.cross(normal1, normal2)

    # Проверяем, параллельны ли плоскости
    if np.linalg.norm(line_direction) < 1e-6:
        print("Плоскости параллельны")
        return None, None

    # находим любую точку на линии пересечения плоскостей
    # Для этого решаем систему уравнений плоскостей
    # A1x + B1y + C1z + d1 = 0, уравнение плоскости
    d1 = -np.dot(normal1, get_point_coord(plane1.corners[0], plane1))
    d2 = -np.dot(normal2, get_point_coord(plane2.corners[0], plane2))

    # Решаем систему уравнений для точки пересечения плоскостей
    # normal1[0]*x + normal1[1]*y + normal1[2]*z = -d1
    # normal2[0]*x + normal2[1]*y + normal2[2]*z = -d2
    A = np.array([normal1, normal2, line_direction])
    b = np.array([-d1, -d2, 0])

    if np.linalg.matrix_rank(A) < 3:
        print("Плоскости не пересекаются в уникальной точке.")
        return None, None

    intersection_point = np.linalg.solve(A[:, :3], b)

    return intersection_point, line_direction


if __name__ == '__main__':
    unittest.main()


class TestFindIntersection(unittest.TestCase):

    def test_simple_intersection(self):
        line1 = np.array([[0, 0], [4, 4]])  # Диагональная линия от (0, 0) до (4, 4)
        line2 = np.array([[0, 4], [4, 0]])  # Диагональная линия от (0, 4) до (4, 0)
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        expected = np.array([2, 2])
        self.assertTrue(np.allclose(intersection, expected), f"Expected {expected}, but got {intersection}")

    def test_parallel_lines(self):
        line1 = np.array([[0, 0], [4, 0]])  # Горизонтальная линия на оси x
        line2 = np.array([[0, 1], [4, 1]])  # Параллельная горизонтальная линия на y = 1
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        self.assertIsNone(intersection, f"Expected None, but got {intersection}")

    def test_intersection_outside_segments(self):
        line1 = np.array([[0, 0], [2, 2]])  # Диагональная линия от (0, 0) до (2, 2)
        line2 = np.array([[3, 3], [5, 5]])  # Диагональная линия от (3, 3) до (5, 5)
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        self.assertIsNone(intersection, f"Expected None, but got {intersection}")

    def test_intersection_on_border(self):
        line1 = np.array([[0, 0], [4, 4]])  # Диагональная линия от (0, 0) до (4, 4)
        line2 = np.array([[4, 4], [6, 2]])  # Диагональная линия от (4, 4) до (6, 2)
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        expected = np.array([4, 4])
        self.assertTrue(np.allclose(intersection, expected), f"Expected {expected}, but got {intersection}")

    def test_perpendicular_intersection(self):
        line1 = np.array([[0, 0], [0, 4]])  # Вертикальная линия от (0, 0) до (0, 4)
        line2 = np.array([[-2, 2], [2, 2]])  # Горизонтальная линия от (-2, 2) до (2, 2)
        intersection = find_intersection_2d(line1, line2[1] - line2[0], line2[0])
        expected = np.array([0, 2])
        self.assertTrue(np.allclose(intersection, expected), f"Expected {expected}, but got {intersection}")


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

        self.assertTrue(np.allclose(local_point, expected_local_point),
                        f"Expected local coordinates {expected_local_point}, but got {local_point}")

    def test_to_global_coord(self):
        edge = np.array([1, 0, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([1, 1, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        local_point = np.array([1, 1])
        global_point = local_system.to_global_coord(local_point)
        expected_global_point = np.array([2, 2, 0])

        self.assertTrue(np.allclose(global_point, expected_global_point),
                        f"Expected global coordinates {expected_global_point}, but got {global_point}")

    def test_round_trip_conversion(self):
        edge = np.array([2, 0, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([3, 3, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        point = np.array([5, 7, 0])
        local_point = local_system.to_local_coord(point)
        global_point = local_system.to_global_coord(local_point)

        self.assertTrue(np.allclose(global_point, point),
                        f"Expected global coordinates {point}, but got {global_point}")

    def test_non_orthogonal_axes(self):
        edge = np.array([1, 1, 0])
        normal = np.array([0, 0, 1])
        origin = np.array([0, 0, 0])
        local_system = LocalSystemCoord(edge, normal, origin)

        point = np.array([2, 2, 0])
        local_point = local_system.to_local_coord(point)
        global_point = local_system.to_global_coord(local_point)

        self.assertTrue(np.allclose(global_point, point),
                        f"Expected global coordinates {point}, but got {global_point}")


class TestFindPointAndLineIntersection(unittest.TestCase):

    def test_simple_intersection(self):
        # Задаем две пересекающиеся плоскости
        plane1 = Plane()
        plane1.corners = np.array([[1, 0, 1], [1, 0, 0], [-1, 0, 1], [-1, 0, 0]])

        plane2 = Plane()
        plane2.corners = np.array([[0, 1, 1], [0, 1, 0], [0, -1, 1], [0, -1, 0]])

        # Ожидаемое направление линии пересечения
        expected_dir = np.array([0, 0, 1])

        # Вызываем функцию для нахождения линии пересечения
        intersection_point, line_direction = get_intersection_line_and_point_of_two_planes(plane1, plane2)

        # Проверяем коллинеарность направления
        line_direction = line_direction / np.linalg.norm(line_direction)
        expected_dir = expected_dir / np.linalg.norm(expected_dir)

        self.assertTrue(np.allclose(line_direction, expected_dir) or np.allclose(line_direction, -expected_dir))

        # Проверяем, что точка лежит на линии пересечения
        self.assertTrue(np.allclose(np.dot(intersection_point, line_direction), 0, atol=1e-6))

    def test_intersection_with_translation(self):
        # Задаем две плоскости
        plane1 = Plane()
        plane1.corners = np.array([[1, 0, 1], [1, 0, 0], [-1, 0, 1], [-1, 0, 0]])

        plane2 = Plane()
        plane2.corners = np.array([[0, 1, 1], [0, 1, 0], [0, -1, 1], [0, -1, 0]])

        # Перемещаем первую плоскость
        plane1.translate(0, 0, 2)

        # Ожидаемое направление линии пересечения
        expected_dir = np.array([0, 0, 1])

        # Вызываем функцию для нахождения линии пересечения
        intersection_point, line_direction = get_intersection_line_and_point_of_two_planes(plane1, plane2)

        # Проверяем коллинеарность направления
        line_direction = line_direction / np.linalg.norm(line_direction)
        expected_dir = expected_dir / np.linalg.norm(expected_dir)

        self.assertTrue(np.allclose(line_direction, expected_dir) or np.allclose(line_direction, -expected_dir))

        # Проверяем, что точка лежит на линии пересечения
        self.assertTrue(np.allclose(np.dot(intersection_point, line_direction), 0, atol=1e-6))

    def test_non_intersecting_planes(self):
        # Задаем две параллельные плоскости
        plane1 = Plane()
        plane1.corners = np.array([[1, 0, 1], [1, 0, 0], [-1, 0, 1], [-1, 0, 0]])

        plane2 = Plane()
        plane2.corners = np.array([[1, 1, 1], [1, 1, 0], [-1, 1, 1], [-1, 1, 0]])

        # Ожидаемый результат: None, т.к. плоскости параллельны
        result = get_intersection_line_and_point_of_two_planes(plane1, plane2)
        self.assertIsNone(result[0])