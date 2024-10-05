import unittest
from unittest.mock import patch, MagicMock
import OpenGL

import numpy as np
import trimesh

from src.node import Node, AABB, HierarchicalNode, Primitive, ObjectWithControlPoints
from src.node import translation, scaling


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = Node()

    def test_initial_state(self):
        """Проверяем начальное состояние узла."""
        self.assertIsInstance(self.node.translation_matrix, np.ndarray)
        self.assertIsInstance(self.node.scaling_matrix, np.ndarray)
        self.assertEqual(self.node.translation_matrix.shape, (4, 4))
        self.assertEqual(self.node.scaling_matrix.shape, (4, 4))
        self.assertFalse(self.node.selected)
        self.assertGreaterEqual(self.node.color_index, 0)
        self.assertLess(self.node.color_index, len(self.node.colors))

    def test_translate(self):
        """Проверяем метод translate."""
        initial_position = np.copy(self.node.translation_matrix)
        self.node.translate(1.0, 2.0, 3.0)
        expected_translation = translation([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(
            self.node.translation_matrix, np.dot(initial_position, expected_translation)
        )

    def test_scale(self):
        """Проверяем метод scale."""
        initial_scaling_matrix = np.copy(self.node.scaling_matrix)
        self.node.scale(up=True)
        expected_scaling = scaling([1.1, 1.1, 1.1])
        np.testing.assert_array_almost_equal(
            self.node.scaling_matrix, np.dot(initial_scaling_matrix, expected_scaling)
        )

    def test_select(self):
        """Проверяем работу выбора узла."""
        self.node.select(True)
        self.assertTrue(self.node.selected)

        self.node.select(False)
        self.assertFalse(self.node.selected)

        self.node.select()
        self.assertTrue(self.node.selected)

        self.node.select()
        self.assertFalse(self.node.selected)

    def test_get_position(self):
        """Проверяем метод получения текущей позиции."""
        self.assertTrue(np.array_equal(self.node.get_position(), [0, 0, 0]))

        self.node.translate(1, 2, 3)
        self.assertTrue(np.array_equal(self.node.get_position(), [1, 2, 3]))

    def test_rotate_color(self):
        """Проверяем смену цвета при вращении."""
        initial_color_index = self.node.color_index

        self.node.rotate_color(forwards=True)
        self.assertEqual(
            self.node.color_index, (initial_color_index + 1) % len(self.node.colors)
        )

        self.node.rotate_color(forwards=False)
        self.assertEqual(self.node.color_index, initial_color_index)

    def test_get_transformed_aabb(self):
        """Проверяем, что трансформированная AABB корректно обновляется."""
        aabb = self.node.get_transformed_aabb()
        self.assertTrue(np.array_equal(aabb.min_point, [0, 0, 0]))
        self.assertTrue(np.array_equal(aabb.max_point, [0.5, 0.5, 0.5]))

        self.node.scale(up=True)
        aabb = self.node.get_transformed_aabb()

        self.assertNotEqual(aabb.max_point.tolist(), [0.5, 0.5, 0.5])


class TestAABB(unittest.TestCase):

    def setUp(self):

        self.min_point = [0.0, 0.0, 0.0]
        self.max_point = [1.0, 1.0, 1.0]
        self.aabb = AABB(self.min_point, self.max_point)

    def test_init(self):

        np.testing.assert_array_equal(self.aabb.min_point, np.array(self.min_point))
        np.testing.assert_array_equal(self.aabb.max_point, np.array(self.max_point))
        expected_extents = np.array(self.max_point) - np.array(self.min_point)
        np.testing.assert_array_equal(self.aabb.original_extents, expected_extents)

        self.assertTrue(isinstance(self.aabb.box, trimesh.primitives.Box))

    def test_ray_hit(self):

        start = [0.5, 0.5, 2.0]
        direction = [0.0, 0.0, -1.0]
        matrix = np.identity(4)

        hit, distance = self.aabb.ray_hit(start, direction, matrix)
        self.assertTrue(hit)
        self.assertAlmostEqual(distance, 1.0, places=5)

        start = [5.0, 5.0, 5.0]
        direction = [0.0, 0.0, 1.0]
        hit, distance = self.aabb.ray_hit(start, direction, matrix)
        self.assertFalse(hit)

    def test_scale(self):

        scale_factor = 2.0
        with patch("trimesh.primitives.Box.apply_scale") as mock_scale:
            self.aabb.scale(scale_factor)
            mock_scale.assert_called_with(scale_factor)

    def test_translate(self):

        translation_vector = [1.0, 1.0, 1.0]
        self.aabb.translate(translation_vector)

        np.testing.assert_array_equal(
            self.aabb.min_point, np.array(self.min_point) + np.array(translation_vector)
        )
        np.testing.assert_array_equal(
            self.aabb.max_point, np.array(self.max_point) + np.array(translation_vector)
        )

    @patch("src.node.glEnable")
    @patch("src.node.glColor3f")
    @patch("src.node.glBegin")
    @patch("src.node.glEnd")
    @patch("src.node.glVertex3fv")
    @patch("src.node.glDisable")
    def test_render(
        self,
        mock_glDisable,
        mock_glVertex3fv,
        mock_glEnd,
        mock_glBegin,
        mock_glColor3f,
        _,
    ):
        self.aabb.render()
        mock_glBegin.assert_called_once()
        mock_glEnd.assert_called_once()
        self.assertGreater(mock_glVertex3fv.call_count, 0)


class TestHierarchicalNode(unittest.TestCase):
    def setUp(self):

        self.node = HierarchicalNode()

    def test_add_child(self):

        child_node = MagicMock()

        self.node.add_child(child_node)

        self.assertIn(child_node, self.node.child_nodes)

    def test_render_self(self):

        child_node1 = MagicMock()
        child_node2 = MagicMock()

        self.node.add_child(child_node1)
        self.node.add_child(child_node2)

        self.node.render_self()

        child_node1.render.assert_called_once()
        child_node2.render.assert_called_once()

    def test_to_dict(self):

        child_node = MagicMock()
        child_node.to_dict.return_value = {"type": "ChildNode"}

        self.node.add_child(child_node)

        result = self.node.to_dict()

        self.assertIn("children", result)
        self.assertEqual(result["children"], [{"type": "ChildNode"}])


class TestPrimitive(unittest.TestCase):
    def setUp(self):

        self.primitive = Primitive()

    @patch("src.node.glCallList")
    def test_render_self(self, mock_glCallList):

        self.primitive.call_list = 123

        self.primitive.render_self()

        mock_glCallList.assert_called_once_with(123)


class TestObjectWithControlPoints(unittest.TestCase):
    def setUp(self):

        self.object_with_cp = ObjectWithControlPoints()
        self.object_with_cp.control_points = [MagicMock(), MagicMock(), MagicMock()]

    @patch("src.node.Node.translate")
    @patch("src.node.get_point_coord")
    @patch("src.premitives.ActivePoint")
    def test_create_control_points(self, mock_ActivePoint, __, _):
        self.object_with_cp.corners = [MagicMock(), MagicMock(), MagicMock()]
        self.object_with_cp.create_control_points()
        self.assertEqual(mock_ActivePoint.call_count, len(self.object_with_cp.corners))

    def test_update_corners(self):
        for point in self.object_with_cp.control_points:
            point.get_position.return_value = [1, 1, 1]

        self.object_with_cp.aabb = None
        self.object_with_cp.update_corners()

        np.testing.assert_array_equal(
            self.object_with_cp.corners, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        )

    def test_translate(self):

        with patch.object(
            ObjectWithControlPoints, "translate", wraps=self.object_with_cp.translate
        ) as mock_translate:

            self.object_with_cp.translate(1, 2, 3)

            mock_translate.assert_called_with(1, 2, 3)

            for point in self.object_with_cp.control_points:
                point.update_position.assert_called_once()


if __name__ == "__main__":
    unittest.main()
