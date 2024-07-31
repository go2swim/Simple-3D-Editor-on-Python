from collections import defaultdict
import numpy as np

from OpenGL.GLUT import *


class Interaction(object):
    def __init__(self):
        self.pressed = None
        self.translation = [0, 0, 0, 0]  # позиция камеры
        self.trackball = Trackball(theta=-25, distance=15)
        self.mouse_loc = None
        self.callbacks = defaultdict(list)

        self.register()

    def register(self):
        glutMouseFunc(self.handle_mouse_button)
        glutMotionFunc(self.handle_mouse_move)
        glutKeyboardFunc(self.handle_keystroke)
        glutSpecialFunc(self.handle_keystroke)

    def register_callback(self, name, func):
        self.callbacks[name].append(func)  # прикрепляем функции из класса viewer к функциям interaction

    def trigger(self, name, *args, **kwargs):
        for func in self.callbacks[name]:
            func(*args, **kwargs)

    def translate(self, x, y, z):
        """ Позиция камеры """
        self.translation[0] += x
        self.translation[1] += y
        self.translation[2] += z

    def handle_mouse_button(self, button, mode, x, y):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        y = ySize - y  # получаем координату для GL
        self.mouse_loc = (x, y)  # сохраняем начальные координаты для отслеживания движения
        if mode == GLUT_DOWN:
            self.pressed = button
            if button == GLUT_RIGHT_BUTTON:
                pass
            elif button == GLUT_LEFT_BUTTON:  # pick
                self.trigger('pick', x, y)
            elif button == 3:  # scroll up
                self.translate(0, 0, 1.0)
            elif button == 4:  # scroll down
                self.translate(0, 0, -1.0)
        else:  # GlUT_UP, не обрабатываем
            self.pressed = None
        glutPostRedisplay()  # обновляем окно

    def handle_mouse_move(self, x, screen_y):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        y = ySize - screen_y  # получаем координату для GL
        if self.pressed is not None:
            dx = x - self.mouse_loc[0]
            dy = y - self.mouse_loc[1]
            if self.pressed == GLUT_RIGHT_BUTTON and self.trackball is not None:
                # при нажатии правой кнопки мыши камера вращается
                self.trackball.drag_to(self.mouse_loc[0], self.mouse_loc[1], dx, dy)
            elif self.pressed == GLUT_LEFT_BUTTON:
                self.trigger('move', x, y)
            elif self.pressed == GLUT_MIDDLE_BUTTON:
                self.translate(dx / 60.0, dy / 60.0, 0)
            else:
                pass
            glutPostRedisplay()
        self.mouse_loc = (x, y)

    def handle_keystroke(self, key, x, screen_y):
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        y = ySize - screen_y
        if key == 's':
            self.trigger('place', 'sphere', x, y)
        elif key == 'c':
            self.trigger('place', 'cube', x, y)
        elif key == GLUT_KEY_UP:
            self.trigger('scale', up=True)
        elif key == GLUT_KEY_DOWN:
            self.trigger('scale', up=False)
        elif key == GLUT_KEY_LEFT:
            self.trigger('rotate_color', forward=True)
        elif key == GLUT_KEY_RIGHT:
            self.trigger('rotate_color', forward=False)
        glutPostRedisplay()


class Trackball:
    def __init__(self, theta=-25, phi=0, distance=15):
        self.theta = theta  # угол вращения вокруг вертикальной оси
        self.phi = phi      # угол наклона вверх/вниз
        self.distance = distance
        self.matrix = np.identity(4)
        self._update_matrix()

    def _update_matrix(self):
        self.matrix = np.identity(4)
        self.matrix[3, 2] = -self.distance
        angle_theta = np.radians(self.theta)
        angle_phi = np.radians(self.phi)
        cos_theta = np.cos(angle_theta)
        sin_theta = np.sin(angle_theta)
        cos_phi = np.cos(angle_phi)
        sin_phi = np.sin(angle_phi)

        # Вращение вокруг вертикальной оси (theta)
        self.matrix[0, 0] = cos_theta
        self.matrix[0, 2] = sin_theta
        self.matrix[2, 0] = -sin_theta
        self.matrix[2, 2] = cos_theta

        # Наклон по горизонтали (phi)
        self.matrix[1, 1] = cos_phi
        self.matrix[1, 2] = -sin_phi
        self.matrix[2, 1] = sin_phi
        self.matrix[2, 2] *= cos_phi

    def drag_to(self, start_x, start_y, delta_x, delta_y):
        self.theta += delta_x * 0.2
        self.phi -= delta_y * 0.2  # изменяем phi при движении мыши по y
        self.phi = max(-90, min(90, self.phi))  # ограничиваем наклон от -90 до 90 градусов
        self._update_matrix()

    def zoom(self, delta):
        self.distance += delta
        self._update_matrix()
