import numpy as np
import unittest

from src.node import get_point_coord
from src.premitives import Plane


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
        self.y_axis = np.cross(normal, self.x_axis) / np.linalg.norm(
            np.cross(normal, self.x_axis)
        )
        self.origin = origin

    def to_local_coord(self, point):
        relative_point = point - self.origin
        x_coord = np.dot(relative_point, self.x_axis)
        y_coord = np.dot(relative_point, self.y_axis)
        return np.array([x_coord, y_coord])

    def to_global_coord(self, point):
        return self.origin + self.x_axis * point[0] + self.y_axis * point[1]


def get_intersection_line_and_point_of_two_planes(plane1, plane2):
    normal1 = np.cross(
        get_point_coord(plane1.corners[1] - plane1.corners[0], plane1),
        get_point_coord(plane1.corners[2] - plane1.corners[0], plane1),
    )
    normal2 = np.cross(
        get_point_coord(plane2.corners[1] - plane2.corners[0], plane2),
        get_point_coord(plane2.corners[2] - plane2.corners[0], plane2),
    )

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
