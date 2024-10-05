import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import json
import os
from datetime import datetime
import numpy as np
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_UNSIGNED_BYTE, GL_RGB
from OpenGL.raw.GLUT import GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT

from src.serialization import (
    save_scene,
    load_scene,
    get_saved_scenes,
    NumpyArrayEncoder,
    load_data,
    export_scene_to_image,
)
from src.scene import Scene
from src.premitives import Cube, Sphere


class TestSerialization(unittest.TestCase):

    @patch("src.serialization.json.dump")
    @patch("src.serialization.open", new_callable=mock_open)
    @patch("src.serialization.datetime")
    def test_save_scene(self, mock_datetime, mock_open_func, mock_json_dump):

        mock_datetime.now.return_value = datetime(2023, 10, 1, 12, 30, 45)

        scene = Scene()
        cube = Cube()
        scene.add_node(cube)

        save_scene(scene)

        mock_open_func.assert_called_once_with(
            "./../data/Save_scene/scene_2023-10-01 12-30-45.json", "w"
        )

        expected_scene_data = {"nodes": [cube.to_dict()]}
        mock_json_dump.assert_called_once_with(
            expected_scene_data, mock_open_func(), indent=4, cls=NumpyArrayEncoder
        )

    @patch("src.serialization.open", new_callable=mock_open, read_data='{"nodes": []}')
    def test_load_scene(self, mock_open_func):

        scene = load_scene("test_scene.json")

        mock_open_func.assert_called_once_with(
            "../data/Save_scene/test_scene.json", "r"
        )

        self.assertIsInstance(scene, Scene)
        self.assertEqual(len(scene.node_list), 0)

    @patch(
        "src.serialization.os.listdir", return_value=["scene_1.json", "scene_2.json"]
    )
    @patch("src.serialization.os.path.isfile", return_value=True)
    def test_get_saved_scenes(self, mock_isfile, mock_listdir):

        saved_scenes = get_saved_scenes()

        expected_scenes = ["scene_1.json", "scene_2.json"]
        self.assertEqual(saved_scenes, expected_scenes)

    def test_numpy_array_encoder(self):

        encoder = NumpyArrayEncoder()
        numpy_array = np.array([1, 2, 3])

        result = json.dumps({"array": numpy_array}, cls=NumpyArrayEncoder)

        self.assertEqual(result, '{"array": [1, 2, 3]}')

    @patch("src.scene.Scene.add_node")
    def test_load_data_with_cube(self, mock_add_node):

        scene_data = {
            "nodes": [{"type": "Cube", "color_index": 2, "position": [1, 1, 1]}]
        }

        scene = load_data(scene_data)

        mock_add_node.assert_called_once()

        added_node = mock_add_node.call_args[0][0]
        self.assertIsInstance(added_node, Cube)

        self.assertEqual(added_node.color_index, 2)

    @patch("src.serialization.glReadBuffer")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("builtins.print")
    @patch("PIL.Image.Image.save")
    @patch("src.serialization.glReadPixels")
    @patch("src.serialization.glutGet")
    @patch("src.serialization.get_name_file_for_save_scene")
    def test_export_scene_to_image(
        self,
        mock_get_name,
        mock_glutGet,
        mock_glReadPixels,
        mock_save,
        mock_print,
        mock_makedirs,
        mock_path_exists,
        _,
    ):

        mock_get_name.return_value = "test_scene"
        mock_glutGet.side_effect = [800, 600]
        mock_glReadPixels.return_value = bytes([255] * (800 * 600 * 3))

        mock_path_exists.return_value = False

        export_scene_to_image()

        mock_glutGet.assert_has_calls(
            [call(GLUT_WINDOW_WIDTH), call(GLUT_WINDOW_HEIGHT)]
        )

        mock_glReadPixels.assert_called_with(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)

        mock_save.assert_called_once()
        saved_image = mock_save.call_args[0][0]
        self.assertTrue(saved_image.endswith(".png"))

        mock_makedirs.assert_called_once()

        mock_print.assert_called_with("Scene in image format saved as test_scene.png")


if __name__ == "__main__":
    unittest.main()
