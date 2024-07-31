from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from matplotlib import colors as mcolors
import numpy as np
import numpy
import random
import trimesh


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
        glMultMatrixf(self.scaling_matrix)

        #cur_color = COLORS[self.color_index]
        glColor3f(*mcolors.to_rgb(self.colors[self.color_index]))
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3]) #цвет собственного излучения материала

        self.render_self() #для каждой фигуры рендер свой

        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0])
        glPopMatrix()

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

class Cube(Primitive):
    def __init__(self):
        super(Cube, self).__init__()
        self.call_list = G_OBJ_CUBE

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
        self.box = trimesh.primitives.Box(
            extents=np.array(max_point) - np.array(min_point),
            transform=trimesh.transformations.translation_matrix(
                (np.array(min_point) + np.array(max_point)) / 2
            )
        )

    def ray_hit(self, start, direction, matrix):
        """
        Проверяет пересечение луча с AABB и возвращает (True, distance) при попадании,
        где distance — расстояние до центра коробки, иначе (False, None).
        """
        # Преобразуем матрицу numpy в формат 4x4, используемый в trimesh
        transformation_matrix = np.array(matrix)

        # Преобразуем луч в локальные координаты AABB
        local_ray_origins = np.dot(trimesh.transformations.inverse_matrix(transformation_matrix),
                                   np.append(start, 1))[:3]
        local_ray_directions = np.dot(transformation_matrix[:3, :3].T, direction)

        # Нормализуем направление луча
        local_ray_directions /= np.linalg.norm(local_ray_directions)

        # Проверяем пересечение луча с AABB
        ray = trimesh.ray.ray_pyembree.RayMeshIntersector(self.box)
        locations, _, _ = ray.intersects_location(
            ray_origins=[local_ray_origins],
            ray_directions=[local_ray_directions],
        )

        if len(locations) > 0:
            # Находим ближайшее пересечение и вычисляем расстояние до центра коробки
            hit_location = locations[0]
            distance = np.linalg.norm(hit_location - self.box.primitive.transform[:3, 3])
            return True, distance
        else:
            return False, None

    def scale(self, scale_factor):
        self.box.apply_scale(scale_factor)



def init_primitives():
    global G_OBJ_SPHERE, G_OBJ_CUBE
    G_OBJ_SPHERE = glGenLists(1)
    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    glutSolidSphere(0.5, 20, 20)
    glEndList()

    G_OBJ_CUBE = glGenLists(1)
    glNewList(G_OBJ_CUBE, GL_COMPILE)
    glutSolidCube(1.0)
    glEndList()