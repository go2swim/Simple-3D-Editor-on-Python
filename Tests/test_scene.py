import unittest
from unittest.mock import MagicMock, patch

from src import node
from src.premitives import Plane, Point, Line, ExtrudedPolygon, Sphere, Cube, SnowFigure
from src.scene import Scene
from src.node import Node, ObjectWithControlPoints, HierarchicalNode
import numpy as np


class TestScene(unittest.TestCase):

    def setUp(self):
        self.scene = Scene()

    def test_init(self):

        self.assertEqual(len(self.scene.node_list), 0)
        self.assertEqual(len(self.scene.select_nodes), 0)

    def test_add_node(self):

        mock_node = MagicMock(spec=Node)
        self.scene.add_node(mock_node)

        self.assertIn(mock_node, self.scene.node_list)

    def test_render(self):

        mock_node1 = MagicMock(spec=Node)
        mock_node2 = MagicMock(spec=Node)
        self.scene.add_node(mock_node1)
        self.scene.add_node(mock_node2)

        self.scene.render()

        mock_node1.render.assert_called_once()
        mock_node2.render.assert_called_once()

    def test_apply_for_each_select_nodes(self):

        mock_node1 = MagicMock(spec=Node)
        mock_node2 = MagicMock(spec=Node)
        self.scene.select_nodes = [mock_node1, mock_node2]

        mock_function = MagicMock()

        self.scene.apply_for_each_select_nodes(mock_function)

        mock_function.assert_any_call(mock_node1)
        mock_function.assert_any_call(mock_node2)

    def test_pick(self):

        mock_node1 = MagicMock(spec=Node)
        mock_node2 = MagicMock(spec=Node)
        self.scene.add_node(mock_node1)
        self.scene.add_node(mock_node2)

        mock_node1.pick.return_value = (True, 5.0)
        mock_node2.pick.return_value = (True, 10.0)

        start = np.array([0, 0, 0])
        direction = np.array([1, 0, 0])
        mat = np.identity(4)

        self.scene.pick(start, direction, mat, multiple_choice=False)

        mock_node1.select.assert_called_once()
        self.assertIn(mock_node1, self.scene.select_nodes)
        self.assertNotIn(mock_node2, self.scene.select_nodes)

        self.assertEqual(mock_node1.depth, 5.0)

        self.scene.pick(start, direction, mat, multiple_choice=True)

    @patch("src.scene.Plane")
    def test_create_plane_from_three_points(self, mock_plane):

        mock_point1 = MagicMock(spec=Point)
        mock_point2 = MagicMock(spec=Point)
        mock_point3 = MagicMock(spec=Point)

        mock_plane.from_three_points.return_value = mock_plane

        mock_plane.control_points = [MagicMock(spec=Point), MagicMock(spec=Point)]

        self.scene.create_plane_from_three_points(mock_point1, mock_point2, mock_point3)

        mock_plane.from_three_points.assert_called_once_with(
            mock_point1, mock_point2, mock_point3
        )

        self.assertIn(mock_plane, self.scene.node_list)
        self.assertIn(mock_plane.control_points[0], self.scene.node_list)
        self.assertIn(mock_plane.control_points[1], self.scene.node_list)

    @patch("src.scene.Scene.create_plane_from_three_points")
    def test_create_plane_from_line_and_point(self, mock_create_three_points):
        mock_line = MagicMock(spec=Line)
        mock_point = MagicMock(spec=Point)

        mock_line.corners = [MagicMock(), MagicMock()]
        node.get_point_coord = MagicMock()

        self.scene.create_plane_from_line_and_point(mock_line, mock_point)

        node.get_point_coord.assert_any_call(mock_line.corners[0], mock_line)
        node.get_point_coord.assert_any_call(mock_line.corners[1], mock_line)

        self.scene.create_plane_from_three_points.assert_called_once()

    @patch("src.scene.Scene.create_plane_from_three_points")
    def test_create_plane_from_plane_and_point(self, mock_create_plane_three_points):

        mock_plane = MagicMock(spec=Plane)
        mock_point = MagicMock(spec=Point)

        mock_plane.corners = [MagicMock(), MagicMock(), MagicMock()]
        node.get_point_coord = MagicMock()

        self.scene.create_plane_from_plane_and_point(mock_plane, mock_point)

        node.get_point_coord.assert_any_call(
            mock_plane.corners[1] - mock_plane.corners[0], mock_plane
        )
        node.get_point_coord.assert_any_call(
            mock_plane.corners[2] - mock_plane.corners[0], mock_plane
        )

        self.scene.create_plane_from_three_points.assert_called_once()

    @patch("src.scene.Line")
    def test_create_line(self, mock_line):
        mock_start = MagicMock()
        mock_end = MagicMock()

        mock_line.control_points = [MagicMock(spec=Point), MagicMock(spec=Point)]

        self.scene.create_line(mock_start, mock_end)

        self.assertNotIn(mock_line, self.scene.node_list)

    def test_dissection_plane(self):
        mock_plane1 = MagicMock(spec=Plane)
        mock_plane2 = MagicMock(spec=Plane)
        self.scene.select_nodes = [mock_plane1, mock_plane2]

        a = [MagicMock(spec=Point), MagicMock(spec=Point)]
        mock_plane1.control_points = MagicMock()
        mock_plane1.control_points.side_effect = [a, []]
        self.scene.node_list = a

        self.scene.dissection_plane()

        for point in mock_plane1.control_points:
            self.assertNotIn(point, self.scene.node_list)

        mock_plane1.intersect_with_plane.assert_called_once_with(mock_plane2)

        for point in mock_plane1.control_points:
            self.assertIn(point, self.scene.node_list)

    @patch("src.scene.ExtrudedPolygon")
    def test_extruded_plane(self, mock_extruded_polygon):

        mock_plane = MagicMock(spec=Plane)
        mock_plane.control_points = []
        self.scene.select_nodes = [mock_plane]
        self.scene.node_list = [mock_plane]

        mock_extruded_polygon.return_value = mock_extruded_polygon

        mock_extruded_polygon.control_points = [
            MagicMock(spec=Point),
            MagicMock(spec=Point),
        ]

        self.scene.extruded_plane()

        mock_extruded_polygon.assert_called_once_with(mock_plane)

        self.assertIn(mock_extruded_polygon, self.scene.node_list)
        self.assertIn(mock_extruded_polygon.control_points[0], self.scene.node_list)
        self.assertIn(mock_extruded_polygon.control_points[1], self.scene.node_list)

        for point in mock_plane.control_points:
            self.assertNotIn(point, self.scene.node_list)

        self.assertNotIn(mock_plane, self.scene.node_list)

        self.assertEqual(len(self.scene.select_nodes), 0)

    def test_scale_selected(self):

        mock_node = MagicMock()
        self.scene.select_nodes = [mock_node]

        self.scene.scale_selected(up=True)

        mock_node.scale.assert_called_once_with(True)

    def test_rotate_selected_color(self):

        mock_node = MagicMock()
        self.scene.select_nodes = [mock_node]

        self.scene.rotate_selected_color(forwards=True)

        mock_node.rotate_color.assert_called_once_with(True)

    def test_delete_selected(self):

        mock_node = MagicMock(spec=ObjectWithControlPoints)
        mock_node.control_points = [MagicMock(), MagicMock()]
        self.scene.select_nodes = [mock_node]

        self.scene.node_list.extend([mock_node] + mock_node.control_points)

        self.scene.delete_selected()

        self.assertNotIn(mock_node, self.scene.node_list)
        for point in mock_node.control_points:
            self.assertNotIn(point, self.scene.node_list)

        self.assertEqual(len(self.scene.select_nodes), 0)

    def test_move_selected(self):

        mock_node = MagicMock()
        mock_node.depth = 1.0
        mock_node.selected_loc = np.array([0.0, 0.0, 0.0])
        self.scene.select_nodes = [mock_node]

        start = np.array([1.0, 1.0, 1.0])
        direction = np.array([0.0, 0.0, 1.0])
        inv_modelview = np.eye(4)

        self.scene.move_selected(start, direction, inv_modelview)

        translation = np.array([0.0, 0.0, 1.0])

        np.testing.assert_array_equal(
            mock_node.selected_loc, start + direction * mock_node.depth
        )

    @patch("src.scene.Scene.delete_selected")
    def test_combine(self, _):

        mock_child1 = MagicMock()
        mock_child2 = MagicMock()
        self.scene.select_nodes = [mock_child1, mock_child2]

        mock_child1.get_transformed_aabb.return_value.min_point = np.array(
            [0.0, 0.0, 0.0]
        )
        mock_child1.get_transformed_aabb.return_value.max_point = np.array(
            [1.0, 1.0, 1.0]
        )
        mock_child2.get_transformed_aabb.return_value.min_point = np.array(
            [1.0, 1.0, 1.0]
        )
        mock_child2.get_transformed_aabb.return_value.max_point = np.array(
            [2.0, 2.0, 2.0]
        )

        new_node = self.scene.combine()

        self.assertIsInstance(new_node, HierarchicalNode)
        self.assertIn(new_node, self.scene.node_list)

        self.assertEqual(new_node.child_nodes, [mock_child1, mock_child2])

        np.testing.assert_array_equal(
            new_node.aabb.min_point, np.array([0.0, 0.0, 0.0])
        )
        np.testing.assert_array_equal(
            new_node.aabb.max_point, np.array([2.0, 2.0, 2.0])
        )

        self.assertEqual(len(self.scene.select_nodes), 2)

    @patch("src.premitives.G_OBJ_SPHERE")
    @patch("src.premitives.G_OBJ_CUBE")
    def test_place(self, _, __):

        inv_modelview = np.eye(4)
        start = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])

        counter = -1
        for shape, expected_class in [
            ("sphere", Sphere),
            ("cube", Cube),
            ("figure", SnowFigure),
            ("point", Point),
        ]:
            self.scene.place(shape, start, direction, inv_modelview)

            counter += 1
            new_node = self.scene.node_list[counter]

            self.assertIsInstance(new_node, expected_class)
            self.assertIn(new_node, self.scene.node_list)

            translation = start + direction * self.scene.PLACE_DEPTH
            new_translation = inv_modelview.dot(
                np.array([translation[0], translation[1], translation[2], 1])
            )


if __name__ == "__main__":
    unittest.main()
