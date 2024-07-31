# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
# import random
#
#
# class Cube:
#     def __init__(self):
#         self.vertices = (
#             (1, -1, -1),
#             (1, 1, -1),
#             (-1, 1, -1),
#             (-1, -1, -1),
#             (1, -1, 1),
#             (1, 1, 1),
#             (-1, -1, 1),
#             (-1, 1, 1)
#         )
#         self.edge = (
#             (0, 1),
#             (0, 3),
#             (0, 4),
#             (2, 1),
#             (2, 3),
#             (2, 7),
#             (6, 3),
#             (6, 4),
#             (6, 7),
#             (5, 1),
#             (5, 4),
#             (5, 7),
#         )
#
#         self.surfaces = (
#             (0, 1, 2, 3),
#             (3, 2, 7, 6),
#             (6, 7, 5, 4),
#             (4, 5, 1, 0),
#             (1, 5, 7, 2),
#             (4, 0, 3, 6)
#         )
#
#     def generate(self):
#         rand_tuple = lambda: (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
#         glBegin(GL_QUADS)
#         for surface in self.surfaces:
#             glColor3fv(rand_tuple())
#             for vertex in surface:
#                 glVertex3fv(self.vertices[vertex])
#         glEnd()
#
#         glBegin(GL_LINES)
#         for edge in self.edge:
#             for vertex in edge:
#                 glVertex3fv(self.vertices[vertex])
#         glEnd()
#
#
# clock = pygame.time.Clock()
#
#
# def main():
#     pygame.init()
#     display = (800, 600)
#     pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
#
#     # задаём усечённую пирамиду (угол обзора, соотношение сторон, ближняя плоскость, дальняя плоскость)
#     gluPerspective(45, display[0] / display[1], 0.5, 50.0)
#     glTranslatef(0., 0., -5.)  # матрица перехода (двигаем камеру)
#
#     cube = Cube()
#
#     while True:
#         clock.tick(60)
#         pygame.display.set_caption(str(int(clock.get_fps())))
#
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         cube.generate()
#         glRotatef(1, 3, 1, 1)
#         pygame.display.flip()
#         pygame.time.wait(10)
#
#
# if __name__ == '__main__':
#     main()
