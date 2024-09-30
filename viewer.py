import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy
from numpy.linalg import norm

from interaction import Interaction
from node import ObjectWithControlPoints, HierarchicalNode
from premitives import init_primitives, Plane, Cube, Sphere, Point
from scene import Scene
import serialization

WINDOW_WIDTH = 480
WINDOW_HEIGHT = 640


class Viewer:
    def __init__(self):
        self._init_interface()
        self.init_opengl()
        init_primitives()
        self.init_grid()
        self.init_scene()
        self.init_interaction()
        self.create_menu()

    def _init_interface(self):
        glutInit()
        glutInitWindowSize(WINDOW_HEIGHT, WINDOW_WIDTH)
        glutInitWindowPosition(50, 50)
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutCreateWindow("3D Editor")
        glutDisplayFunc(self.render)

    def init_opengl(self):
        self.inverseModelView = numpy.identity(4)
        self.modelView = numpy.identity(4)

        glEnable(GL_CULL_FACE) #убираем фигуры имеющие определёный порядок обхода
        glCullFace(GL_BACK) #отбраковываем невидимые фигуры
        glEnable(GL_DEPTH_TEST) #включаем z буфер
        glDepthFunc(GL_LESS) #задаёт параметр теста(отбрасываем дальние)

        #включаем источник света
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 0, 1, 0)) #задаём позицию
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1)) #указываем что это прожектор и задаём направление


        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.4, 0.4, 0.4, 0.0)

    def init_scene(self):
        self.load_scene()

    def create_sample_scene(self):
        def create_plane(corners_coords):
            plane = Plane()
            plane.corners = corners_coords
            self.scene.add_node(plane)
            plane.create_control_points()
            for control_point in plane.control_points:
                self.scene.add_node(control_point)

        cube_node = Cube()
        cube_node.translate(-7, 0.5, 0)
        cube_node.color_index = 2
        self.scene.add_node(cube_node)

        sphere_node = Sphere()
        sphere_node.translate(-5, 0.5, 0)
        sphere_node.color_index = 3
        self.scene.add_node(sphere_node)

        create_plane(np.array([[-3, 1, 0], [-3, 0, 0], [-2, 1, 0], [-2, 0, 0]]))

        # пересекающиеся плоскости
        self.scene.create_plane_from_three_points([1., 1., 0.], [1., 0., 0.], [-2., 0., 0.])
        self.scene.create_plane_from_three_points([0., -1., 0.], [0., 1., 0.], [0., 1., 1.])

        create_plane(np.array([[2, 0, -1], [2, 0, 1], [3, 0, -1], [3, 0, 1]]))

        self.scene.create_line([6, 0, 0], [6, 2, 0])
        point = Point()
        point.translate(6.2, 1, 1)
        self.scene.add_node(point)

        point1 = Point()
        point2 = Point()
        point3 = Point()
        point1.translate(8, 1, 0)
        point2.translate(9, 1, 1)
        point3.translate(9, 2, 1)
        self.scene.add_node(point1)
        self.scene.add_node(point2)
        self.scene.add_node(point3)

        create_plane(np.array([[10, 1, 0], [10, 0, 0], [11, 1, 0], [11, 0, 0]]))
        point = Point()
        point.translate(10.5, 0.5, 1)
        self.scene.add_node(point)

        create_plane(np.array([[10, 1, 0], [10, 0, 0], [11, 1, 0], [11, 0, 0]]))


    def init_interaction(self):
        """ Привязка функций обработки вызовов от interaction """
        self.interaction = Interaction()
        self.interaction.register_callback('pick', self.pick)
        self.interaction.register_callback('move', self.move)
        self.interaction.register_callback('place', self.place)
        self.interaction.register_callback('rotate_color', self.rotate_color)
        self.interaction.register_callback('scale', self.scale)
        self.interaction.register_callback('delete', self.delete)
        self.interaction.register_callback('multiple_choice', self.multiple_choice)
        self.interaction.register_callback('combine', self.combine)
        self.interaction.register_callback('create_menu', self.create_menu)
        self.interaction.register_callback('dissection', self.dissection_plane)
        self.interaction.register_callback('extrude', self.extrude_plane)
        self.interaction.register_callback('save', self.save_scene)
        self.interaction.register_callback('load', self.load_scene)

    def save_scene(self):
        serialization.save_scene(self.scene)
        self.create_menu()

    def load_scene(self, filename="Demonstration_scene.json"):
        self.scene = serialization.load_scene(filename)
        # self.scene = Scene()
        # self.create_sample_scene()

    def main_loop(self):
        glutMainLoop()

    def render(self):
        self.init_view()

        glEnable(GL_LIGHTING)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #меняем положение трекбола
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        loc = self.interaction.translation
        glTranslated(loc[0], loc[1], loc[2]) #двигаем трекбол
        glMultMatrixf(self.interaction.trackball.matrix) #подгружаем текущую матрицу трекбола

        #сохраняем
        currentModelView = numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        self.modelView = numpy.transpose(currentModelView)
        self.inverseModelView = numpy.linalg.inv(numpy.transpose(currentModelView))

        # рендерим каждый объект на сцене
        self.scene.render()

        # отрисовка сетки
        glDisable(GL_LIGHTING) #отключаем свет чтобы она выделялась
        glCallList(G_OBJ_PLANE)
        glPopMatrix()

        # ждём очистки буферов, чтобы начать отрисовку сцены
        glFlush()


    def init_view(self):
        #параметры экрана
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        aspect_ratio = float(xSize) / float(ySize)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glViewport(0, 0, xSize, ySize) #задаёт преобразование из нормальных в экранные координаты
        gluPerspective(70, aspect_ratio, 0.1, 1000.0) #задаём усечённую пирамиду
        glTranslated(0, 0, -15) #двигаем камеру

    def get_ray(self, x, y):
        """ Генерация луча """
        self.init_view()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # get two points on the line.
        start = numpy.array(gluUnProject(x, y, 0.001))
        end = numpy.array(gluUnProject(x, y, 0.999))

        # convert those points into a ray
        direction = end - start
        direction = direction / norm(direction)

        return start, direction

    def dissection_plane(self):
        self.scene.dissection_plane()

    def extrude_plane(self):
        self.scene.extruded_plane()

    # методы обработки событий из interaction
    def pick(self, x, y, multiple_choice=False):
        start, direction = self.get_ray(x, y)
        self.scene.pick(start, direction, self.modelView, multiple_choice)

    def multiple_choice(self, x, y):
        # print(f'condition ctrl is: {bool(x)}')
        self.pick(x, y, True)

    def combine(self):
        self.scene.combine()

    def move(self, x, y):
        """ Execute a move command on the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.move_selected(start, direction, self.inverseModelView)

    def rotate_color(self, forward):
        """
        Rotate the color of the selected Node.
        Boolean 'forward' indicates direction of rotation.
        """
        self.scene.rotate_selected_color(forward)

    def scale(self, up):
        """ Scale the selected Node. Boolean up indicates scaling larger."""
        self.scene.scale_selected(up)

    def place(self, shape, x, y):
        """ Execute a placement of a new primitive into the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.place(shape, start, direction, self.inverseModelView)

    def delete(self):
        self.scene.delete_selected()

    def create_menu(self):
        """Создание вложенного меню для средней кнопки мыши."""

        files = serialization.get_saved_scenes()

        # Для каждого файла создаем пункт в меню
        load2_menu = glutCreateMenu(self.menu_select)
        for index, file_name in enumerate(files):
            glutAddMenuEntry(file_name.replace('.json', ''), 100 + index)

        # Создаем дочернее меню для загрузки/сохранения сцен
        load_menu = glutCreateMenu(self.menu_select)
        glutAddMenuEntry("Save scene (K)", 1)
        glutAddMenuEntry("Create new scene", 3)
        glutAddSubMenu("Load scene (L)", load2_menu)

        create_menu = glutCreateMenu(self.menu_select)
        glutAddMenuEntry("Point (P)", 4)
        glutAddMenuEntry("Cube (C)", 5)
        glutAddMenuEntry("Sphere (S)", 6)

        change_menu = glutCreateMenu(self.menu_select)
        glutAddMenuEntry("Delete (Del)", 7)
        glutAddMenuEntry("Next color (<-, ->)", 8)
        glutAddMenuEntry("Scale up (Up)", 9)
        glutAddMenuEntry("Scale down (Down)", 10)

        action_with_selected_menu = glutCreateMenu(self.menu_select)
        glutAddMenuEntry("Dissection plane (R)", 11)
        glutAddMenuEntry("Extrude plane (Q)", 12)

        # Создаем основное меню и добавляем в него дочерние
        main_menu = glutCreateMenu(self.menu_select)
        glutAddSubMenu("Scene manager (L)", load_menu)
        glutAddSubMenu("Create", create_menu)
        glutAddSubMenu("Change", change_menu)
        glutAddSubMenu("Action with selected", action_with_selected_menu)

        # Привязываем меню к средней кнопке мыши
        glutAttachMenu(GLUT_MIDDLE_BUTTON)

    def menu_select(self, value):
        """Обработка выбора пункта меню."""
        center_of_window = (WINDOW_HEIGHT / 2, WINDOW_WIDTH / 2)

        if value == 1:
            self.save_scene()
        elif value == 3:
            self.scene = Scene()  # создание новой сцены
        elif value == 4:
            self.place('point', center_of_window[0], center_of_window[1])
        elif value == 5:
            self.place('cube', center_of_window[0], center_of_window[1])
        elif value == 6:
            self.place('sphere', center_of_window[0], center_of_window[1])
        elif value == 7:
            self.delete()
        elif value == 8:
            self.rotate_color(forward=True)
        elif value == 9:
            self.scale(up=True)
        elif value == 10:
            self.scale(up=False)
        elif value == 11:
            self.dissection_plane()
        elif value == 12:
            self.extrude_plane()
        elif value >= 100:
            files = serialization.get_saved_scenes()
            try:
                filename = files[value - 100]
                self.load_scene(filename)
            except IndexError:
                print('Файла ент')
                raise IndexError

        glutPostRedisplay()
        return 0

    def init_grid(self):
        global G_OBJ_PLANE
        G_OBJ_PLANE = glGenLists(1)
        glNewList(G_OBJ_PLANE, GL_COMPILE)
        self.draw_grid()
        glEndList()

    def draw_grid(self, size=50, step=1):
        glBegin(GL_LINES)
        glColor3f(0.8, 0.8, 0.8)  # Цвет сетки (серый)
        for i in range(-size, size + 1, step):
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
        glEnd()


if __name__ == '__main__':
    viewer = Viewer()
    viewer.main_loop()