import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from OpenGL.raw.GL.VERSION.GL_1_0 import (
    GL_CULL_FACE,
    GL_BACK,
    GL_DEPTH_TEST,
    GL_LESS,
    GL_LIGHT0,
    GL_POSITION,
    GL_SPOT_DIRECTION,
    GL_FRONT_AND_BACK,
    GL_COLOR_MATERIAL,
    GL_AMBIENT_AND_DIFFUSE,
    GL_LIGHTING,
    GL_COLOR_BUFFER_BIT,
    GL_MODELVIEW,
    GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION,
    GL_COMPILE,
    GL_LINES,
)
from OpenGL.raw.GL._types import GLfloat_4, GLfloat_3
from OpenGL.raw.GLUT import GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT, GLUT_MIDDLE_BUTTON

from viewer import Viewer, WINDOW_WIDTH, WINDOW_HEIGHT


class TestViewer(unittest.TestCase):

    @patch.object(Viewer, "_init_interface")
    @patch.object(Viewer, "init_opengl")
    @patch("viewer.init_primitives")
    @patch.object(Viewer, "init_grid")
    @patch.object(Viewer, "init_scene")
    @patch.object(Viewer, "init_interaction")
    @patch.object(Viewer, "create_menu")
    def setUp(
        self,
        mock_create_menu,
        mock_init_interaction,
        mock_init_scene,
        mock_init_grid,
        mock_init_primitives,
        mock_init_opengl,
        mock_init_interface,
    ):

        self.mock_create_menu = mock_create_menu
        self.mock_init_interaction = mock_init_interaction
        self.mock_init_scene = mock_init_scene
        self.mock_init_grid = mock_init_grid
        self.mock_init_primitives = mock_init_primitives
        self.mock_init_opengl = mock_init_opengl
        self.mock_init_interface = mock_init_interface

        self.viewer = Viewer()

    def test_init_calls(self):
        self.mock_init_interface.assert_called_once()
        self.mock_init_primitives.assert_called_once()
        self.mock_init_grid.assert_called_once()
        self.mock_init_scene.assert_called_once()
        self.mock_init_interaction.assert_called_once()
        self.mock_create_menu.assert_called_once()

    @patch("viewer.glutCreateWindow")
    @patch("viewer.glutDisplayFunc")
    @patch("viewer.glutInitDisplayMode")
    @patch("viewer.glutInitWindowPosition")
    @patch("viewer.glutInitWindowSize")
    @patch("viewer.glutInit")
    def test_init_interface(
        self,
        mock_glutInit,
        mock_glutInitWindowSize,
        mock_glutInitWindowPosition,
        mock_glutInitDisplayMode,
        mock_glutCreateWindow,
        mock_glutDisplayFunc,
    ):
        self.viewer._init_interface()

        mock_glutInit.assert_called_once()
        mock_glutInitWindowSize.assert_called_once_with(640, 480)
        mock_glutInitWindowPosition.assert_called_once_with(50, 50)
        mock_glutInitDisplayMode.assert_called_once()
        mock_glutCreateWindow.assert_called_once_with(self.viewer.render)
        mock_glutDisplayFunc.assert_called_once_with("3D Editor")

    @patch("viewer.glCullFace")
    @patch("viewer.glDepthFunc")
    @patch("viewer.glClearColor")
    @patch("viewer.glLightfv")
    @patch("viewer.glEnable")
    @patch("viewer.glColorMaterial")
    def test_init_opengl(
        self,
        mock_glColorMaterial,
        mock_glEnable,
        mock_glLightfv,
        mock_glClearColor,
        mock_glDepthFunc,
        mock_glCullFace,
    ):
        self.viewer.init_opengl()

        mock_glEnable.assert_any_call(GL_CULL_FACE)
        mock_glCullFace.assert_called_once_with(GL_BACK)
        mock_glEnable.assert_any_call(GL_DEPTH_TEST)
        mock_glDepthFunc.assert_called_once_with(GL_LESS)
        mock_glEnable.assert_any_call(GL_LIGHT0)
        mock_glColorMaterial.assert_called_once_with(
            GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE
        )
        mock_glEnable.assert_any_call(GL_COLOR_MATERIAL)
        mock_glClearColor.assert_called_once_with(0.4, 0.4, 0.4, 0.0)

    @patch("viewer.serialization.load_scene")
    def test_init_scene(self, mock_load_scene):
        self.viewer.init_scene()
        mock_load_scene.assert_called_once()

    @patch("viewer.Plane")
    @patch("src.premitives.ActivePoint")
    @patch("viewer.Point")
    @patch("viewer.Sphere")
    @patch("viewer.Cube")
    @patch.object(Viewer, "scene", create=True)
    def test_create_sample_scene(
        self,
        mock_scene,
        mock_cube,
        mock_sphere,
        mock_point,
        mock_active_point,
        mock_plane,
    ):
        mock_scene.add_node = MagicMock()
        mock_scene.create_plane_from_three_points = MagicMock()
        mock_scene.create_line = MagicMock()

        self.viewer.create_sample_scene()

        self.assertEqual(mock_scene.add_node.call_count, 11)
        self.assertEqual(mock_scene.create_plane_from_three_points.call_count, 2)
        mock_scene.create_line.assert_called_once_with([6, 0, 0], [6, 2, 0])
        mock_cube.assert_called_once()
        mock_sphere.assert_called_once()
        self.assertEqual(mock_point.call_count, 5)

    @patch("viewer.Interaction")
    def test_init_interaction(self, mock_init_interaction):
        self.viewer.interaction = MagicMock()
        mock_init_interaction.return_value = self.viewer.interaction

        self.viewer.init_interaction()

        expected_callbacks = [
            "pick",
            "move",
            "place",
            "rotate_color",
            "scale",
            "delete",
            "multiple_choice",
            "combine",
            "create_menu",
        ]

        print(mock_init_interaction.register_callback.call_list())
        for callback in expected_callbacks:
            self.viewer.interaction.register_callback.assert_any_call(
                callback, getattr(self.viewer, callback)
            )

    @patch("viewer.serialization.save_scene")
    @patch.object(Viewer, "create_menu")
    def test_save_scene(self, mock_create_menu, mock_save_scene):
        self.viewer.scene = MagicMock()
        self.viewer.save_scene()
        mock_save_scene.assert_called_once_with(self.viewer.scene)
        mock_create_menu.assert_called_once()

    @patch("viewer.serialization.load_scene")
    def test_load_scene(self, mock_load_scene):
        self.viewer.load_scene()
        mock_load_scene.assert_called_once_with("Demonstration_scene.json")

    @patch("viewer.glutMainLoop")
    def test_main_loop(self, mock_glutMainLoop):
        self.viewer.main_loop()
        mock_glutMainLoop.assert_called_once()

    @patch.object(Viewer, "init_view")
    @patch.object(Viewer, "draw_grid")
    @patch("viewer.glNewList")
    @patch("viewer.glEndList")
    @patch("viewer.glGenLists")
    @patch("viewer.glCallList")
    @patch("viewer.glEnable")
    @patch("viewer.glDisable")
    @patch("viewer.glClear")
    @patch("viewer.glMatrixMode")
    @patch("viewer.glPushMatrix")
    @patch("viewer.glLoadIdentity")
    @patch("viewer.glTranslated")
    @patch("viewer.glMultMatrixf")
    @patch("viewer.glGetFloatv")
    @patch("viewer.glPopMatrix")
    @patch("viewer.glFlush")
    @patch.object(Viewer, "scene", create=True)
    def test_render(
        self,
        mock_scene,
        mock_glFlush,
        mock_glPopMatrix,
        mock_glGetFloatv,
        mock_glMultMatrixf,
        mock_glTranslated,
        mock_glLoadIdentity,
        mock_glPushMatrix,
        mock_glMatrixMode,
        mock_glClear,
        mock_glDisable,
        mock_glEnable,
        mock_init_view,
        _,
        __,
        ___,
        ____,
        _____,
    ):

        mock_get_floatv_result = np.eye(4)
        mock_glGetFloatv.return_value = mock_get_floatv_result

        self.viewer.init_grid()
        self.viewer.interaction = MagicMock()
        self.viewer.render()

        mock_glEnable.assert_called_once_with(GL_LIGHTING)
        mock_glClear.assert_called_once_with(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        mock_glMatrixMode.assert_any_call(GL_MODELVIEW)
        mock_glPushMatrix.assert_called_once()
        mock_glLoadIdentity.assert_called_once()

        mock_glTranslated.assert_called_once_with(
            self.viewer.interaction.translation[0],
            self.viewer.interaction.translation[1],
            self.viewer.interaction.translation[2],
        )
        mock_glMultMatrixf.assert_called_once_with(
            self.viewer.interaction.trackball.matrix
        )
        mock_glGetFloatv.assert_called_once()
        mock_scene.render.assert_called_once()
        mock_glPopMatrix.assert_called_once()
        mock_glFlush.assert_called_once()

    @patch("viewer.glMatrixMode")
    @patch("viewer.glLoadIdentity")
    @patch("viewer.glViewport")
    @patch("viewer.gluPerspective")
    @patch("viewer.glTranslated")
    @patch("viewer.glutGet")
    def test_init_view(
        self,
        mock_glutGet,
        mock_glTranslated,
        mock_gluPerspective,
        mock_glViewport,
        mock_glLoadIdentity,
        mock_glMatrixMode,
    ):

        mock_glutGet.side_effect = [800, 600]

        self.viewer.init_view()

        mock_glutGet.assert_any_call(GLUT_WINDOW_WIDTH)
        mock_glutGet.assert_any_call(GLUT_WINDOW_HEIGHT)
        mock_glMatrixMode.assert_called_once_with(GL_PROJECTION)
        mock_glLoadIdentity.assert_called_once()
        mock_glViewport.assert_called_once_with(0, 0, 800, 600)
        mock_gluPerspective.assert_called_once_with(70, 800 / 600, 0.1, 1000.0)
        mock_glTranslated.assert_called_once_with(0, 0, -15)

    @patch.object(Viewer, "init_view")
    @patch("viewer.glMatrixMode")
    @patch("viewer.glLoadIdentity")
    @patch("viewer.gluUnProject")
    def test_get_ray(self, mock_gluUnProject, _, __, ___):
        mock_gluUnProject.side_effect = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        start, direction = self.viewer.get_ray(100, 200)

        self.assertTrue(np.array_equal(start, np.array([1, 2, 3])))
        self.assertTrue(
            np.array_equal(direction, np.array([3, 3, 3]) / np.linalg.norm([3, 3, 3]))
        )
        self.assertEqual(mock_gluUnProject.call_count, 2)

    def test_dissection_plane(self):
        self.viewer.scene = MagicMock()
        self.viewer.dissection_plane()
        self.viewer.scene.dissection_plane.assert_called_once()

    def test_extrude_plane(self):
        self.viewer.scene = MagicMock()
        self.viewer.extrude_plane()
        self.viewer.scene.extruded_plane.assert_called_once()

    @patch(
        "viewer.Viewer.get_ray", return_value=(np.array([1, 2, 3]), np.array([4, 5, 6]))
    )
    def test_pick(self, _):
        self.viewer.scene = MagicMock()
        self.viewer.modelView = MagicMock()
        self.viewer.pick(100, 200)

        self.viewer.scene.pick.assert_called_once()

    @patch.object(
        Viewer,
        "get_ray",
        return_value=(
            np.array([1, 2, 3]),
            np.array([3, 3, 3]) / np.linalg.norm([3, 3, 3]),
        ),
    )
    def test_move(self, mock_get_ray):
        self.viewer.inverseModelView = MagicMock()
        self.viewer.scene = MagicMock()

        self.viewer.move(100, 200)

        mock_get_ray.assert_called_once_with(100, 200)

    def test_rotate_color(self):
        self.viewer.scene = MagicMock()
        self.viewer.rotate_color(True)
        self.viewer.scene.rotate_selected_color.assert_called_once_with(True)

    def test_scale(self):
        self.viewer.scene = MagicMock()
        self.viewer.scale(True)
        self.viewer.scene.scale_selected.assert_called_once_with(True)

    @patch(
        "viewer.Viewer.get_ray", return_value=(np.array([1, 2, 3]), np.array([4, 5, 6]))
    )
    @patch("viewer.gluUnProject")
    def test_place(self, mock_gluUnProject, _):
        mock_gluUnProject.side_effect = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        self.viewer.scene = MagicMock()
        self.viewer.inverseModelView = MagicMock()

        self.viewer.place("Cube", 100, 200)

        self.viewer.scene.place.assert_called_once()

    def test_delete(self):
        self.viewer.scene = MagicMock()
        self.viewer.delete()
        self.viewer.scene.delete_selected.assert_called_once()

    @patch("viewer.glutCreateMenu")
    @patch("viewer.glutAddMenuEntry")
    @patch("viewer.glutAddSubMenu")
    @patch("viewer.glutAttachMenu")
    @patch("src.serialization.get_saved_scenes")
    def test_create_menu(
        self,
        mock_get_saved_scenes,
        mock_glutAttachMenu,
        mock_glutAddSubMenu,
        mock_glutAddMenuEntry,
        mock_glutCreateMenu,
    ):
        """Тестирование создания меню"""
        mock_get_saved_scenes.return_value = ["scene1.json", "scene2.json"]

        self.viewer.create_menu()

        self.assertEqual(mock_glutCreateMenu.call_count, 6)
        mock_glutAddMenuEntry.assert_any_call("scene1", 100)
        mock_glutAddMenuEntry.assert_any_call("scene2", 101)
        mock_glutAttachMenu.assert_called_once_with(GLUT_MIDDLE_BUTTON)

    @patch("viewer.glutPostRedisplay")
    @patch("src.serialization.get_saved_scenes")
    def test_menu_select(self, mock_get_saved_scenes, mock_glutPostRedisplay):
        """Тестирование выбора пункта меню"""
        mock_get_saved_scenes.return_value = ["scene1.json", "scene2.json"]

        with patch.object(self.viewer, "save_scene") as mock_save_scene:
            self.viewer.menu_select(1)
            mock_save_scene.assert_called_once()

        with patch.object(self.viewer, "place") as mock_place:
            self.viewer.menu_select(4)
            mock_place.assert_called_once_with(
                "point", WINDOW_HEIGHT / 2, WINDOW_WIDTH / 2
            )

        with patch.object(self.viewer, "rotate_color") as mock_rotate_color:
            self.viewer.menu_select(8)
            mock_rotate_color.assert_called_once_with(forward=True)

        with patch.object(self.viewer, "load_scene") as mock_load_scene:
            self.viewer.menu_select(100)
            mock_load_scene.assert_called_once_with("scene1.json")

        with patch("builtins.print") as mock_print:
            try:
                self.viewer.menu_select(102)
            except IndexError:
                pass
            mock_print.assert_called_with("Файла ент")

        mock_glutPostRedisplay.assert_called()

    @patch("viewer.glGenLists")
    @patch("viewer.glNewList")
    @patch("viewer.glEndList")
    @patch.object(Viewer, "draw_grid")
    def test_init_grid(
        self, mock_draw_grid, mock_glEndList, mock_glNewList, mock_glGenLists
    ):
        """Тестирование инициализации сетки"""
        mock_glGenLists.return_value = 1

        self.viewer.init_grid()

        mock_glGenLists.assert_called_once_with(1)
        mock_glNewList.assert_called_once_with(1, GL_COMPILE)
        mock_draw_grid.assert_called_once()
        mock_glEndList.assert_called_once()

    @patch("viewer.glColor3f")
    @patch("viewer.glBegin")
    @patch("viewer.glVertex3f")
    @patch("viewer.glEnd")
    def test_draw_grid(self, mock_glEnd, mock_glVertex3f, mock_glBegin, _):
        """Тестирование отрисовки сетки"""
        self.viewer.draw_grid(size=2, step=1)

        mock_glBegin.assert_called_once_with(GL_LINES)
        self.assertEqual(mock_glVertex3f.call_count, 20)
        mock_glEnd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
