import unittest
from unittest.mock import MagicMock, patch

import src.interaction
from OpenGL.raw.GLUT import (
    GLUT_LEFT_BUTTON,
    GLUT_DOWN,
    GLUT_KEY_UP,
    GLUT_KEY_LEFT,
    GLUT_RIGHT_BUTTON,
)

from src.interaction import Interaction, Trackball


class TestInteraction(unittest.TestCase):

    @patch("src.interaction.glutMouseFunc")
    @patch("src.interaction.glutMotionFunc")
    @patch("src.interaction.glutKeyboardFunc")
    @patch("src.interaction.glutSpecialFunc")
    def setUp(self, _, __, ___, ____):
        self.interaction = Interaction()

    def test_register_callback(self):
        mock_callback = MagicMock()

        self.interaction.register_callback("pick", mock_callback)

        self.interaction.trigger("pick", 10, 20)

        mock_callback.assert_called_once_with(10, 20)

    def test_translate(self):
        self.interaction.translate(1, 2, 3)

        self.assertEqual(self.interaction.translation, [1, 2, 3, 0])

    @patch("src.interaction.glutPostRedisplay")
    @patch("src.interaction.Interaction.translate")
    @patch("src.interaction.glutGet", return_value=800)
    def test_handle_mouse_button_left_pick(
        self, mock_glutGet, mock_translate, mock_redisplay
    ):
        mock_callback = MagicMock()
        self.interaction.register_callback("pick", mock_callback)

        self.interaction.handle_mouse_button(4, GLUT_DOWN, 100, 200)

        mock_translate.assert_called_once_with(0, 0, -1.0)
        mock_redisplay.assert_called_once()

    @patch("src.interaction.glutPostRedisplay")
    @patch("src.interaction.glutGet", return_value=800)
    def test_handle_mouse_button_right_zoom(self, mock_glutGet, _):
        self.interaction.handle_mouse_button(3, GLUT_DOWN, 100, 200)

        self.assertEqual(self.interaction.translation[2], 1.0)

        self.interaction.handle_mouse_button(4, GLUT_DOWN, 100, 200)

        self.assertEqual(self.interaction.translation[2], 0.0)

    @patch("src.interaction.glutPostRedisplay")
    @patch("src.interaction.glutGet", return_value=800)
    def test_handle_mouse_move(self, mock_glutGet, _):

        self.interaction.trackball = MagicMock()

        self.interaction.pressed = GLUT_RIGHT_BUTTON
        self.interaction.mouse_loc = (100, 100)
        self.interaction.handle_mouse_move(150, 200)

        self.interaction.trackball.drag_to.assert_called_once_with(100, 100, 50, 500)

    @patch("src.interaction.glutPostRedisplay")
    @patch("src.interaction.glutGet", side_effect=[100, 200, 100, 200])
    def test_handle_keystroke(self, _, __):
        mock_callback_save = MagicMock()
        mock_callback_load = MagicMock()

        self.interaction.register_callback("save", mock_callback_save)
        self.interaction.register_callback("extrude", mock_callback_load)

        self.interaction.handle_keystroke(b"k", 0, 0)

        mock_callback_save.assert_called_once()

        self.interaction.handle_keystroke(b"q", 0, 0)

        mock_callback_load.assert_called_once()

    @patch("src.interaction.glutPostRedisplay")
    @patch("src.interaction.glutGet", side_effect=[100, 200, 100, 200])
    def test_handle_special_keystroke(self, _, __):
        mock_callback_scale = MagicMock()
        mock_callback_rotate_color = MagicMock()

        self.interaction.register_callback("scale", mock_callback_scale)
        self.interaction.register_callback("rotate_color", mock_callback_rotate_color)

        self.interaction.handle_special_keystroke(GLUT_KEY_UP, 0, 0)

        mock_callback_scale.assert_called_once_with(up=True)

        self.interaction.handle_special_keystroke(GLUT_KEY_LEFT, 0, 0)

        mock_callback_rotate_color.assert_called_once_with(forward=True)

    def test_trackball_drag_to(self):
        trackball = Trackball()

        self.assertEqual(trackball.theta, -25)
        self.assertEqual(trackball.phi, 0)

        trackball.drag_to(0, 0, 100, 50)

        self.assertNotEqual(trackball.theta, -25)
        self.assertNotEqual(trackball.phi, 0)

    def test_trackball_zoom(self):
        trackball = Trackball()

        self.assertEqual(trackball.distance, 15)

        trackball.zoom(5)

        self.assertEqual(trackball.distance, 20)


if __name__ == "__main__":
    unittest.main()
