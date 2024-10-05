import numpy
import numpy as np
import pyrr.ray
import trimesh
from OpenGL.GL import (
    glEnable,
    glPopMatrix,
    glPushMatrix,
    glMultMatrixf,
    glBegin,
    glEnd,
    glDisable,
    glGenLists,
    GL_CULL_FACE,
    GL_LINES,
    glVertex3fv,
)
from OpenGL.raw.GL.VERSION.GL_1_0 import (
    glIsEnabled,
    GL_TRIANGLE_STRIP,
    glColor3f,
    glMaterialfv,
    GL_FRONT,
    GL_EMISSION,
    glNewList,
    glEndList,
    GL_COMPILE,
)
from OpenGL.raw.GLUT import glutSolidSphere, glutSolidCube
from matplotlib import colors as mcolors

from src.node import (
    Primitive,
    AABB,
    translation,
    Node,
    get_point_coord,
    HierarchicalNode,
    scaling,
    ObjectWithControlPoints,
)

G_OBJ_POINT = None
G_OBJ_SPHERE = None
G_OBJ_CUBE = None


class Point(Primitive):
    def __init__(self):
        super().__init__()
        self.call_list = G_OBJ_POINT
        self.aabb = AABB([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])

    def scale(self, up):
        return


class Sphere(Primitive):
    def __init__(self):
        super(Sphere, self).__init__()
        self.call_list = G_OBJ_SPHERE
        self.aabb = AABB([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])


class Cube(Primitive):
    def __init__(self):
        super(Cube, self).__init__()
        self.call_list = G_OBJ_CUBE
        self.aabb = AABB([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])


class SnowFigure(HierarchicalNode):
    def __init__(self):
        super(SnowFigure, self).__init__()
        self.child_nodes = [Sphere(), Sphere(), Sphere()]
        self.child_nodes[0].translate(0, -0.6, 0)  # scale 1.0
        self.child_nodes[1].translate(0, 0.1, 0)
        self.child_nodes[1].scaling_matrix = numpy.dot(
            self.scaling_matrix, scaling([0.8, 0.8, 0.8])
        )
        self.child_nodes[2].translate(0, 0.75, 0)
        self.child_nodes[2].scaling_matrix = numpy.dot(
            self.scaling_matrix, scaling([0.7, 0.7, 0.7])
        )
        for child_node in self.child_nodes:
            child_node.color_index = 0
        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 1.1, 0.5])


class ActivePoint(Point):
    def __init__(self, parent_object, position=np.array([0, 0, 0])):
        super().__init__()
        self.parent_object = parent_object
        self.translation_matrix = translation(position)
        self.scaling_matrix = self.parent_object.scaling_matrix

    def translate(self, x, y, z):
        super().translate(x, y, z)
        self.parent_object.update_corners()

    def update_position(self):
        """Обновляем позицию точки на основе матриц трансформации плоскости"""
        corner_idx = self.parent_object.control_points.index(self)
        transformed_corner = (
            self.parent_object.translation_matrix
            @ self.parent_object.scaling_matrix
            @ np.append(self.parent_object.corners[corner_idx], 1)
        )[:3]
        Node.translate(self, *transformed_corner - self.get_position())


class Plane(ObjectWithControlPoints):
    def __init__(self):
        super(Plane, self).__init__()
        self.corners = None
        self.control_points = list()
        self.aabb = None  # коллизия в этом классе определяется по другому
        self.points = list()  # точки касаний, для отладки
        self.lines = list()

    def calculate_corners(self):
        """Вычисляет угловые точки прямоугольной плоскости."""
        if self.corners is not None:
            return self.corners

        # Если углы не заданы, вернём пустой массив
        return np.zeros((4, 3))

    @classmethod
    def from_three_points(cls, p1, p2, p3, scale_factor=1):
        """Создаёт плоскость по трём точкам."""
        p1, p2, p3 = map(np.array, (p1, p2, p3))

        # Вычисляем векторы по двум сторонам треугольника
        v1 = p2 - p1
        v2 = p3 - p1

        # Нормаль к плоскости
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Центр треугольника, используем как центр плоскости
        center = (p1 + p2 + p3) / 3

        # Выбираем вектор, перпендикулярный нормали
        tangent = np.cross(normal, (1, 0, 0))
        if np.allclose(tangent, 0):
            tangent = np.cross(normal, (0, 1, 0))
        tangent /= np.linalg.norm(tangent)

        bitangent = np.cross(normal, tangent)
        bitangent /= np.linalg.norm(bitangent)

        width = np.linalg.norm(p2 - p1) * scale_factor
        height = np.linalg.norm(p3 - (p1 + p2) / 2) * scale_factor

        # Создаём экземпляр плоскости
        plane = cls()

        # Определяем угловые точки прямоугольника
        half_width = width / 2
        half_height = height / 2
        plane.corners = np.array(
            [
                center
                + tangent * -half_width
                + bitangent * half_height,  # Верхний левый угол
                center
                + tangent * half_width
                + bitangent * half_height,  # Верхний правый угол
                center
                + tangent * -half_width
                + bitangent * -half_height,  # Нижний левый угол
                center
                + tangent * half_width
                + bitangent * -half_height,  # Нижний правый угол
            ]
        )

        plane.create_control_points()

        print(plane.corners)
        v1 = plane.corners[0] - plane.corners[1]
        v2 = plane.corners[0] - plane.corners[2]
        v3 = plane.corners[0] - plane.corners[3]
        print(f"volume of parallelepiped: {np.cross(v1, v2) @ v3}")
        return plane

    def render_self(self):
        if self.corners is None:
            return

        cull_face_enabled = glIsEnabled(GL_CULL_FACE)
        glDisable(GL_CULL_FACE)

        glBegin(GL_TRIANGLE_STRIP)
        for corner in self.corners:
            glVertex3fv(corner)
        glEnd()

        if cull_face_enabled:
            glEnable(GL_CULL_FACE)

    def pick(self, start, direction, matrix):
        """Проверка пересечения луча с плоскостью"""
        # Преобразуем начальную точку и направление луча в локальную систему координат
        transformation_matrix = np.array(matrix)
        start_local = np.dot(
            trimesh.transformations.inverse_matrix(transformation_matrix),
            np.append(start, 1),
        )[:3]
        direction_local = np.dot(transformation_matrix[:3, :3].T, direction)
        direction_local /= np.linalg.norm(direction_local)

        # Определяем плоскость
        plane_normal = np.cross(
            get_point_coord(self.corners[1] - self.corners[0], self),
            get_point_coord(self.corners[2] - self.corners[0], self),
        )
        plane_point = np.dot(self.translation_matrix, np.append(self.corners[0], 1))[:3]
        plane_obj = pyrr.plane.create_from_position(
            position=plane_point, normal=plane_normal
        )

        # Создаём луч
        ray_obj = pyrr.ray.create(start=start_local, direction=direction_local)

        # Находим точку пересечения
        intersect_point = pyrr.geometric_tests.ray_intersect_plane(ray_obj, plane_obj)

        if intersect_point is None:
            # Плоскость находится за лучом
            return False, None

        # point = Point()
        # point.translate(*intersect_point)
        # self.points.append(point)
        # print(f'intersection point: {intersect_point}')

        center = (self.corners[0] - self.corners[1]) / 2 + (
            self.corners[0] - self.corners[2]
        ) / 2

        if self.is_point_inside(intersect_point):
            return True, np.linalg.norm(start - intersect_point)
        return False, None

    def get_corner_coord(self, corner):
        return ((self.translation_matrix) @ self.scaling_matrix @ np.append(corner, 1))[
            :3
        ]

    def is_point_inside(self, point):
        # Преобразуем углы в двумерное пространство плоскости
        edge1 = self.get_corner_coord(self.corners[1]) - self.get_corner_coord(
            self.corners[0]
        )
        edge2 = self.get_corner_coord(self.corners[2]) - self.get_corner_coord(
            self.corners[0]
        )
        point_vector = point - self.get_corner_coord(self.corners[0])

        u = np.dot(point_vector, edge1) / (np.dot(edge1, edge1))
        v = np.dot(point_vector, edge2) / (np.dot(edge2, edge2))
        # print(f'local coord: {u}, {v}')

        return 0 <= u <= 1 and 0 <= v <= 1

    def render(self):
        # убран рендер aabb
        glPushMatrix()

        # for point in self.points:
        #     point.render()
        #
        # for lines in self.lines:
        #     lines.render()

        glMultMatrixf(
            numpy.transpose(self.translation_matrix)
        )  # переводим объект в ск камеры
        glMultMatrixf(self.scaling_matrix)

        glColor3f(*mcolors.to_rgb(self.colors[self.color_index]))
        if self.selected:
            glMaterialfv(
                GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3]
            )  # цвет собственного излучения материала

        self.render_self()  # для каждой фигуры рендер свой

        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0])
        glPopMatrix()

    def intersect_with_plane(self, other_plane):
        from src.intersections import (
            get_intersection_line_and_point_of_two_planes,
            LocalSystemCoord,
            find_intersection_2d,
            point_on_line,
        )

        result = get_intersection_line_and_point_of_two_planes(self, other_plane)
        if result is None:
            return
        intersection_point, line_direction = result

        point = Point()
        point.translate(*intersection_point)
        self.points.append(point)

        line = Line(intersection_point, intersection_point + line_direction)
        self.lines.append(line)

        corner0 = get_point_coord(self.corners[0], self)
        corner1 = get_point_coord(self.corners[1], self)

        new_basis = LocalSystemCoord(
            corner1 - corner0, np.cross(corner0, corner1), corner0
        )

        local_intersection_point = new_basis.to_local_coord(intersection_point)
        local_line_direction = new_basis.to_local_coord(line_direction)

        # обновляем угловые точки новой плоскости
        # проходимся по кругу по рёбрам и находим точки пересечения на них
        new_corners = [[] for _ in range(len(self.corners))]
        nodes = [0, 2, 3, 1]  # правильная последовательность обхода углов плоскости
        for i in range(4):
            local_start = new_basis.to_local_coord(
                get_point_coord(self.corners[nodes[i]], self)
            )
            local_end = new_basis.to_local_coord(
                get_point_coord(self.corners[nodes[(i + 1) % 4]], self)
            )

            intersection_point = find_intersection_2d(
                np.array([local_start, local_end]),
                local_line_direction,
                local_intersection_point,
            )

            if point_on_line(intersection_point, np.array([local_start, local_end])):
                intersect_point_world = new_basis.to_global_coord(intersection_point)

                if nodes[i] == 2 or nodes[i] == 0:
                    new_corners[nodes[i + 1]] = intersect_point_world
                else:
                    new_corners[nodes[i]] = intersect_point_world

                point = Point()
                point.translate(*intersect_point_world)
                self.points.append(point)
                print(f"intersect line point: {intersect_point_world}")

        print(new_corners)
        for i in range(len(new_corners)):
            if len(new_corners[nodes[i]]) == 0:
                new_corners[nodes[i]] = get_point_coord(self.corners[nodes[i]], self)

        self.control_points.clear()
        self.corners = new_corners
        self.translation_matrix = np.identity(4)
        self.scaling_matrix = np.identity(4)
        self.create_control_points()


class Line(ObjectWithControlPoints):
    def __init__(self, start, end):
        super(Line, self).__init__()
        self.corners = [np.array(start, float), np.array(end, float)]
        self.control_points = [ActivePoint(self, start), ActivePoint(self, end)]
        self.update_aabb()

    def update_aabb(self):
        """Обновление AABB на основе текущих точек линии и их AABB."""
        def get_transform_corner_point(control_point, func):
            return (
                control_point.scaling_matrix
                @ control_point.translation_matrix
                @ np.append(func(control_point.aabb), 1)
            )[:3]

        min = lambda aabb: aabb.min_point
        max = lambda aabb: aabb.max_point

        min_point = np.minimum(
            *(
                (get_transform_corner_point(self.control_points[i], max))
                for i in range(2)
            )
        )
        max_point = np.maximum(
            *(
                (get_transform_corner_point(self.control_points[i], min))
                for i in range(2)
            )
        )

        # Задаём небольшой отступ для удобства выбора
        padding = np.array([0.1, 0.1, 0.1])

        min_point += padding
        max_point -= padding

        self.aabb = AABB(min_point, max_point)

    def render_self(self):
        """Рендер линии."""
        glBegin(GL_LINES)
        glVertex3fv(self.corners[0])
        glVertex3fv(self.corners[1])
        glEnd()

    def get_position(self):
        return (self.corners[0] + self.corners[1]) / 2

    def update_corners(self):
        """Обновляет углы плоскости на основе текущих позиций точек-контроллеров."""
        self.corners = np.array([point.get_position() for point in self.control_points])
        self.translation_matrix = np.identity(4)
        self.scaling_matrix = np.identity(4)
        self.update_aabb()


class ExtrudedPolygon(ObjectWithControlPoints):
    def __init__(self, base_plane, extrusion_height=1.0):
        super(ExtrudedPolygon, self).__init__()
        self.create_corners(extrusion_height, base_plane)
        self.create_control_points()
        self.aabb = None
        # Список плоскостей (4 боковые, 1 верхняя, 1 нижняя)
        self.planes = []
        self.update_planes()

    def update_planes(self):
        """Создаёт плоскости для многогранника."""
        self.planes.clear()

        nodes = [0, 2, 3, 1]
        # Создаем боковые плоскости
        for k, i in enumerate(nodes):
            j = nodes[(k + 1) % len(nodes)]
            quad_corners = [
                get_point_coord(self.corners[i], self),
                get_point_coord(self.corners[j], self),
                get_point_coord(self.corners[4 + i], self),
                get_point_coord(self.corners[4 + j], self),
            ]
            plane = Plane()
            plane.corners = quad_corners
            self.planes.append(plane)

        # Создаем верхнюю и нижнюю плоскости
        plane1 = Plane()
        plane2 = Plane()
        plane1.corners = [get_point_coord(corner, self) for corner in self.corners[:4]]
        plane2.corners = [get_point_coord(corner, self) for corner in self.corners[4:]]
        self.planes.append(plane1)
        self.planes.append(plane2)

    def create_corners(self, extrusion_height, base_plane):
        base_vertices = [
            get_point_coord(corner, base_plane) for corner in base_plane.corners
        ]
        top_vertices = list(range(4))
        nodes = [0, 2, 3, 1]
        for i, node in enumerate(nodes):
            normal = np.cross(
                base_vertices[nodes[(i + 1) % 4]] - base_vertices[node],
                base_vertices[nodes[(i - 1) % 4]] - base_vertices[node],
            )
            normal /= np.linalg.norm(normal)
            top_vertices[node] = base_vertices[node] + normal * extrusion_height * -1

        self.corners = list(base_vertices) + top_vertices

    def update_corners(self):
        super().update_corners()
        self.update_planes()

    def render_self(self):
        """Рендерит многогранник."""
        # Рендерим плоскости
        for plane in self.planes:
            plane.render_self()

        # Рендерим грани (белые линии)
        glColor3f(1.0, 1.0, 1.0)

        glBegin(GL_LINES)
        # Рендерим боковые грани
        for i in range(4):
            base_corner = get_point_coord(self.corners[i], self)
            top_corner = get_point_coord(self.corners[4 + i], self)
            glVertex3fv(base_corner)
            glVertex3fv(top_corner)

        # Рендерим грани основания и верхней грани
        nodes = [0, 2, 3, 1]
        for k, i in enumerate(nodes):
            j = nodes[(k + 1) % len(nodes)]

            # Основание
            glVertex3fv(get_point_coord(self.corners[i], self))
            glVertex3fv(get_point_coord(self.corners[j], self))

            # Верхняя грань
            glVertex3fv(get_point_coord(self.corners[4 + i], self))
            glVertex3fv(get_point_coord(self.corners[4 + j], self))

        glEnd()

    def pick(self, start, direction, mat):
        """Проверка пересечения луча с многогранником."""
        hit_any = False
        closest_distance = float("inf")
        closest_hit_point = None

        for plane in self.planes:
            hit, distance = plane.pick(start, direction, mat)
            if hit and distance < closest_distance:
                closest_distance = distance
                closest_hit_point = hit
                hit_any = True

        return hit_any, closest_distance if hit_any else None

    def translate(self, x, y, z):
        super().translate(x, y, z)
        self.corners = [get_point_coord(corner, self) for corner in self.corners]
        self.translation_matrix = np.identity(4)
        self.scaling_matrix = np.identity(4)
        self.update_planes()


def init_primitives():
    global G_OBJ_SPHERE, G_OBJ_CUBE, G_OBJ_POINT
    G_OBJ_SPHERE = glGenLists(1)
    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    glutSolidSphere(0.5, 20, 20)  # радиус, количество линий по ширине и долготе
    glEndList()

    G_OBJ_CUBE = glGenLists(1)
    glNewList(G_OBJ_CUBE, GL_COMPILE)
    glutSolidCube(1.0)
    glEndList()

    G_OBJ_POINT = glGenLists(1)
    glNewList(G_OBJ_POINT, GL_COMPILE)
    glutSolidSphere(0.08, 20, 20)
    glEndList()
