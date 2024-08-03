from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from matplotlib import colors as mcolors
import numpy as np
import numpy
import random
import trimesh
from trimesh import Trimesh


class Node(object):
    """ Base class for scene elements """
    def __init__(self):
        self.colors = list(mcolors.XKCD_COLORS.values())
        self.color_index = random.randint(0, len(self.colors))
        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])  # задаём "колайдер" узла
        self.translation_matrix = numpy.identity(4)
        self.scaling_matrix = numpy.identity(4)
        self.selected = False

    def render(self):
        glPushMatrix()
        glMultMatrixf(numpy.transpose(self.translation_matrix)) #переводим объект в ск камеры
        self.aabb.render()
        glMultMatrixf(self.scaling_matrix)

        glColor3f(*mcolors.to_rgb(self.colors[self.color_index]))
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3]) #цвет собственного излучения материала

        self.render_self() #для каждой фигуры рендер свой

        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0])
        glPopMatrix()

    def get_transformed_aabb(self):
        """ Возвращает AABB с учетом всех применённых трансформаций,
            нужно для построения нового aabb для объеденной фигуры """
        # Вычисляем матрицу трансформации, которая включает перевод и масштабирование
        transform = numpy.dot(self.translation_matrix, self.scaling_matrix)

        # Применяем трансформацию к углам AABB
        transformed_min_point = np.dot(transform, np.append(self.aabb.min_point, 1))[:3]
        transformed_max_point = np.dot(transform, np.append(self.aabb.max_point, 1))[:3]

        # Формируем новый AABB с учётом всех преобразований
        return AABB(
            np.minimum(transformed_min_point, transformed_max_point),
            np.maximum(transformed_min_point, transformed_max_point)
        )

    def pick(self, start, direction, mat):
        """ Проверка луча на касание с node """

        # transform the modelview matrix by the current translation
        newmat = numpy.dot(
            numpy.dot(mat, self.translation_matrix),
            numpy.linalg.inv(self.scaling_matrix)
        )
        results = self.aabb.ray_hit(start, direction, newmat)  # проверяем касается ли с AABB
        return results

    def select(self, select=None):
        """ Toggles or sets selected state """
        if select is not None:
            self.selected = select
        else:
            self.selected = not self.selected

    def render_self(self):
        raise NotImplementedError(
            "The Abstract Node Class doesn't define 'render_self'")

    def scale(self, up):
        s = 1.1 if up else 0.9
        self.scaling_matrix = numpy.dot(self.scaling_matrix, scaling([s, s, s]))
        self.aabb.scale(s)

    def translate(self, x, y, z):
        self.translation_matrix = numpy.dot(
            self.translation_matrix,
            translation([x, y, z]))

    def rotate_color(self, forwards):
        self.color_index += 1 if forwards else -1
        self.color_index %= len(self.colors)


def scaling(scale):
    """ Возвращает матрицу для увелечения scale """
    s = numpy.identity(4)
    s[0, 0] = scale[0]
    s[1, 1] = scale[1]
    s[2, 2] = scale[2]
    s[3, 3] = 1
    return s

def translation(displacement):
    t = numpy.identity(4)
    t[0, 3] = displacement[0]
    t[1, 3] = displacement[1]
    t[2, 3] = displacement[2]
    return t


class HierarchicalNode(Node):
    def __init__(self):
        super(HierarchicalNode, self).__init__()
        self.child_nodes = []

    def render_self(self):
        for child in self.child_nodes:
            child.render()

class Primitive(Node):
    def __init__(self):
        super(Primitive, self).__init__()
        self.call_list = None

    def render_self(self):
        glCallList(self.call_list)

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
        self.child_nodes[0].translate(0, -0.6, 0) # scale 1.0
        self.child_nodes[1].translate(0, 0.1, 0)
        self.child_nodes[1].scaling_matrix = numpy.dot(
            self.scaling_matrix, scaling([0.8, 0.8, 0.8]))
        self.child_nodes[2].translate(0, 0.75, 0)
        self.child_nodes[2].scaling_matrix = numpy.dot(
            self.scaling_matrix, scaling([0.7, 0.7, 0.7]))
        for child_node in self.child_nodes:
            child_node.color_index = 0
        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 1.1, 0.5])


class AABB:
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)
        self.original_extents = self.max_point - self.min_point
        self.box = trimesh.primitives.Box(
            extents=self.original_extents,
            transform=trimesh.transformations.translation_matrix(
                (self.min_point + self.max_point) / 2
            )
        )
        self.transform = trimesh.transformations.translation_matrix(
            (self.min_point + self.max_point) / 2
        )

    def ray_hit(self, start, direction, matrix):
        """ Проверяет пересечение луча с AABB """
        transformation_matrix = np.array(matrix)
        local_ray_origins = np.dot(trimesh.transformations.inverse_matrix(transformation_matrix),
                                   np.append(start, 1))[:3]
        local_ray_directions = np.dot(transformation_matrix[:3, :3].T, direction)
        local_ray_directions /= np.linalg.norm(local_ray_directions)

        ray = trimesh.ray.ray_pyembree.RayMeshIntersector(self.box)
        locations, _, _ = ray.intersects_location(
            ray_origins=[local_ray_origins],
            ray_directions=[local_ray_directions],
        )

        if len(locations) > 0:
            hit_location = locations[0]
            distance = np.linalg.norm(start - hit_location)
            print(f'pick {self.box.__str__()}')
            return True, distance
        else:
            return False, None

    def scale(self, scale_factor):
        """ Масштабирует AABB и обновляет коллайдер """
        self.box.apply_scale(scale_factor)
        print(self.box.scale)

    def translate(self, translation_vector):
        """ Перемещает AABB на заданный вектор """
        translation_matrix = trimesh.transformations.translation_matrix(translation_vector)
        self.transform = trimesh.transformations.concatenate_matrices(self.transform, translation_matrix)
        self.box.apply_transform(translation_matrix)
        self.min_point += translation_vector
        self.max_point += translation_vector

    def render(self):
        """ Рендерит грани AABB """
        corners = self.box.vertices
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Нижняя грань
            [4, 5], [5, 6], [6, 7], [7, 4],  # Верхняя грань
            [0, 4], [1, 5], [2, 6], [3, 7]   # Вертикальные ребра
        ]

        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)
        glBegin(GL_LINES)
        for edge in edges:
            glVertex3fv(corners[edge[0]])
            glVertex3fv(corners[edge[1]])
        glEnd()
        glEnable(GL_LIGHTING)



def init_primitives():
    global G_OBJ_SPHERE, G_OBJ_CUBE, G_OBJ_POINT
    G_OBJ_SPHERE = glGenLists(1)
    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    glutSolidSphere(0.5, 20, 20) #радиус, количество линий по ширине и долготе
    glEndList()

    G_OBJ_CUBE = glGenLists(1)
    glNewList(G_OBJ_CUBE, GL_COMPILE)
    glutSolidCube(1.0)
    glEndList()

    G_OBJ_POINT = glGenLists(1)
    glNewList(G_OBJ_POINT, GL_COMPILE)
    glutSolidSphere(0.08, 20, 20)
    glEndList()


class Point(Primitive):
    def __init__(self):
        super().__init__()
        self.call_list = G_OBJ_POINT
        self.aabb = AABB([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])

    def scale(self, up):
        return