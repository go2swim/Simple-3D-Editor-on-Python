from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy
from numpy.linalg import norm

from interaction import Interaction
from node import Cube, Sphere, SnowFigure, init_primitives
from scene import Scene


class Viewer:
    def __init__(self):
        self._init_interface()
        self.init_opengl()
        init_primitives()
        self.init_grid()
        self.init_scene()
        self.init_interaction()

    def _init_interface(self):
        glutInit()
        glutInitWindowSize(640, 480)
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


        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) #хз
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.4, 0.4, 0.4, 0.0)

    def init_scene(self):
        self.scene = Scene()
        self.create_sample_scene()

    def create_sample_scene(self):
        cube_node = Cube()
        cube_node.translate(2, 0, 2)
        cube_node.color_index = 2
        self.scene.add_node(cube_node)

        sphere_node = Sphere()
        sphere_node.translate(-2, 0, 2)
        sphere_node.color_index = 3
        self.scene.add_node(sphere_node)

        hierarchical_node = SnowFigure()
        hierarchical_node.translate(-2, 0, -2)
        self.scene.add_node(hierarchical_node)

    def init_interaction(self):
        """ Привязка функций обработки вызовов от interaction """
        self.interaction = Interaction()
        self.interaction.register_callback('pick', self.pick)
        self.interaction.register_callback('move', self.move)
        self.interaction.register_callback('place', self.place)
        self.interaction.register_callback('rotate_color', self.rotate_color)
        self.interaction.register_callback('scale', self.scale)

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

    def pick(self, x, y):
        start, direction = self.get_ray(x, y)
        self.scene.pick(start, direction, self.modelView)

    # методы обработки событий из interaction
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