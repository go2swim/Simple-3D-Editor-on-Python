import unittest
import numpy as np
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_TRIANGLE_STRIP, GL_LINES

from src.premitives import (
    Point,
    AABB,
    Sphere,
    Cube,
    SnowFigure,
    init_primitives,
    ActivePoint,
    Plane,
    Line,
    ExtrudedPolygon,
)
from unittest.mock import patch, MagicMock, Mock
import src.premitives


class TestSnowFigure(unittest.TestCase):

    @patch("src.premitives.Sphere")
    def setUp(self, mock_sphere):
        self.snow_figure = SnowFigure()
        self.mock_sphere = mock_sphere

    def test_initial_child_nodes(self):
        self.assertEqual(len(self.snow_figure.child_nodes), 3)
        self.assertIsInstance(self.snow_figure.child_nodes[0], MagicMock)
        self.assertIsInstance(self.snow_figure.child_nodes[1], MagicMock)
        self.assertIsInstance(self.snow_figure.child_nodes[2], MagicMock)

    def test_translation_and_scaling(self):
        self.assertTrue(
            np.allclose(self.snow_figure.child_nodes[0].translation_matrix[1, 3], -0.6)
        )
        self.assertTrue(
            np.allclose(self.snow_figure.child_nodes[2].scaling_matrix[0, 0], 0.7)
        )


class TestActivePoint(unittest.TestCase):

    @patch("src.premitives.Point.__init__")
    def setUp(self, _):
        self.parent_object = Mock()
        self.active_point = ActivePoint(self.parent_object)

    def test_translate(self):
        self.active_point.translate(1, 1, 1)
        self.parent_object.update_corners.assert_called_once()

    def test_update_position(self):
        self.parent_object.control_points = [self.active_point]
        self.parent_object.translation_matrix = np.identity(4)
        self.parent_object.scaling_matrix = np.identity(4)
        self.parent_object.corners = [np.array([1, 1, 1])]

        self.active_point.update_position()
        position = self.active_point.get_position()

        self.assertTrue(np.allclose(position, [1, 1, 1]))


class TestPlane(unittest.TestCase):
    def setUp(self):
        self.plane = Plane()
        self.plane = Plane()
        self.plane.corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.plane.translation_matrix = np.identity(4)
        self.plane.scaling_matrix = np.identity(4)
        self.plane.colors = ["blue", "green", "red"]
        self.plane.color_index = 0
        self.plane.selected = False
        self.plane.points = []
        self.plane.lines = []

    def test_calculate_corners_no_corners(self):
        """Тест, когда угловые точки не заданы."""
        self.plane.corners = None
        result = self.plane.calculate_corners()
        self.assertTrue(np.array_equal(result, np.zeros((4, 3))))

    def test_calculate_corners_with_corners(self):
        """Тест, когда угловые точки уже заданы."""
        expected_corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.plane.corners = expected_corners
        result = self.plane.calculate_corners()
        self.assertTrue(np.array_equal(result, expected_corners))

    @patch("src.node.ObjectWithControlPoints.create_control_points")
    @patch("src.premitives.ActivePoint")
    def test_from_three_points(self, _, mock_control_points):
        """Тест создания плоскости по трём точкам."""
        p1 = [0.0, 0, 0]
        p2 = [1.0, 0, 0]
        p3 = [0.0, 1, 0]
        plane = Plane.from_three_points(p1, p2, p3, scale_factor=1)

        expected_corners = np.array(
            [[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
        )
        self.assertFalse(np.allclose(plane.corners, expected_corners))
        mock_control_points.assert_called_once()

    @patch("src.premitives.glIsEnabled")
    @patch("src.premitives.glDisable")
    @patch("src.premitives.glEnable")
    @patch("src.premitives.glBegin")
    @patch("src.premitives.glEnd")
    @patch("src.premitives.glVertex3fv")
    def test_render_self(self, mock_glVertex3fv, mock_glEnd, mock_glBegin, _, __, ___):
        """Тест отрисовки углов плоскости."""
        self.plane.corners = np.array(
            [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [1.0, 1, 0]]
        )

        self.plane.render_self()

        mock_glBegin.assert_called_once_with(GL_TRIANGLE_STRIP)
        self.assertEqual(mock_glVertex3fv.call_count, 4)
        mock_glEnd.assert_called_once()

    @patch("pyrr.geometric_tests.ray_intersect_plane")
    @patch("pyrr.plane.create_from_position")
    @patch("trimesh.transformations.inverse_matrix")
    def test_pick(self, mock_inverse_matrix, mock_create_plane, mock_ray_intersect):
        """Тест на пересечение луча с плоскостью."""
        start = np.array([0.5, 0.5, 2])
        direction = np.array([0, 0, -1])
        matrix = np.identity(4)

        mock_inverse_matrix.return_value = np.identity(4)
        mock_create_plane.return_value = MagicMock()
        mock_ray_intersect.return_value = np.array([0.5, 0.5, 0])

        hit, distance = self.plane.pick(start, direction, matrix)

        self.assertTrue(hit)
        self.assertAlmostEqual(distance, 2)

    def test_get_corner_coord(self):
        """Тест для получения координат угла с учётом матриц трансформации."""
        corner = np.array([1, 0, 0])
        expected_coord = np.array([1, 0, 0])

        result = self.plane.get_corner_coord(corner)

        self.assertTrue(np.allclose(result, expected_coord))

    def test_is_point_inside(self):
        """Тест на проверку, находится ли точка внутри плоскости."""
        point_inside = np.array([0.5, 0.5, 0])
        point_outside = np.array([1.5, 1.5, 0])

        self.assertTrue(self.plane.is_point_inside(point_inside))

        self.assertFalse(self.plane.is_point_inside(point_outside))

    @patch("src.premitives.glPushMatrix")
    @patch("src.premitives.glPopMatrix")
    @patch("src.premitives.glMultMatrixf")
    @patch("src.premitives.glColor3f")
    @patch("src.premitives.glMaterialfv")
    @patch("matplotlib.colors.to_rgb")
    def test_render(
        self,
        mock_to_rgb,
        mock_glMaterialfv,
        mock_glColor3f,
        mock_glMultMatrixf,
        mock_glPopMatrix,
        mock_glPushMatrix,
    ):
        """Тест для метода render, который рендерит объект на сцене с использованием OpenGL."""
        mock_to_rgb.return_value = [0.0, 0.0, 1.0]
        self.plane.render_self = MagicMock()

        self.plane.render()

        mock_glPushMatrix.assert_called_once()
        mock_glPopMatrix.assert_called_once()
        mock_glColor3f.assert_called_once_with(0.0, 0.0, 1.0)
        self.plane.render_self.assert_called_once()

    @patch("src.node.ObjectWithControlPoints.create_control_points")
    @patch("src.node.Node.translate")
    @patch("src.premitives.G_OBJ_POINT")
    @patch("src.intersections.get_intersection_line_and_point_of_two_planes")
    @patch("src.intersections.LocalSystemCoord")
    @patch("src.intersections.find_intersection_2d")
    @patch("src.intersections.point_on_line")
    def test_intersect_with_plane(
        self,
        mock_point_on_line,
        mock_find_intersection_2d,
        mock_LocalSystemCoord,
        mock_get_intersection_line_and_point_of_two_planes,
        _,
        __,
        mock_control_points,
    ):
        """Тест для метода intersect_with_plane, который находит линию пересечения двух плоскостей."""
        mock_get_intersection_line_and_point_of_two_planes.return_value = (
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
        )
        mock_LocalSystemCoord.return_value.to_local_coord = MagicMock(
            side_effect=lambda x: x
        )
        mock_LocalSystemCoord.return_value.to_global_coord = MagicMock(
            side_effect=lambda x: x
        )
        mock_find_intersection_2d.return_value = np.array([0.5, 0.5])
        mock_point_on_line.return_value = True

        other_plane = MagicMock()
        self.plane.corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        self.plane.intersect_with_plane(other_plane)

        self.assertEqual(len(self.plane.points), 5)
        self.assertEqual(len(self.plane.lines), 1)
        self.assertEqual(len(self.plane.corners), 4)
        mock_control_points.assert_called_once()


class TestLine(unittest.TestCase):
    def setUp(self):
        """Инициализация линии с двумя точками для тестов."""
        self.start_point = [0, 0, 0]
        self.end_point = [1, 1, 1]
        self.line = Line(self.start_point, self.end_point)

        self.line.control_points[0].aabb = MagicMock(
            min_point=np.array([0, 0, 0]), max_point=np.array([1, 1, 1])
        )
        self.line.control_points[1].aabb = MagicMock(
            min_point=np.array([1, 1, 1]), max_point=np.array([2, 2, 2])
        )
        self.line.control_points[0].translation_matrix = np.identity(4)
        self.line.control_points[1].translation_matrix = np.identity(4)
        self.line.control_points[0].scaling_matrix = np.identity(4)
        self.line.control_points[1].scaling_matrix = np.identity(4)

    @patch("src.premitives.glBegin")
    @patch("src.premitives.glEnd")
    @patch("src.premitives.glVertex3fv")
    def test_render_self(self, mock_glVertex3fv, mock_glEnd, mock_glBegin):
        """Тест метода render_self для рендеринга линии с помощью OpenGL."""

        self.line.render_self()

        mock_glBegin.assert_called_once_with(GL_LINES)
        mock_glEnd.assert_called_once()

    def test_update_aabb(self):
        """Тест метода update_aabb для обновления AABB на основе текущих точек."""

        self.line.update_aabb()

        expected_min_point = np.array([1.1, 1.1, 1.1])
        expected_max_point = np.array([0.9, 0.9, 0.9])

        np.testing.assert_array_almost_equal(
            self.line.aabb.min_point, expected_min_point
        )
        np.testing.assert_array_almost_equal(
            self.line.aabb.max_point, expected_max_point
        )

    def test_get_position(self):
        """Тест метода get_position для вычисления средней точки линии."""

        position = self.line.get_position()

        expected_position = (np.array(self.start_point) + np.array(self.end_point)) / 2
        np.testing.assert_array_almost_equal(position, expected_position)

    def test_update_corners(self):
        """Тест метода update_corners для обновления углов линии."""

        self.line.control_points[0].get_position = MagicMock(
            return_value=np.array([0, 0, 0])
        )
        self.line.control_points[1].get_position = MagicMock(
            return_value=np.array([1, 1, 1])
        )

        self.line.update_corners()

        expected_corners = np.array([[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_almost_equal(self.line.corners, expected_corners)

        np.testing.assert_array_almost_equal(
            self.line.translation_matrix, np.identity(4)
        )
        np.testing.assert_array_almost_equal(self.line.scaling_matrix, np.identity(4))


class TestExtrudedPolygon(unittest.TestCase):

    def setUp(self):
        """Создание тестового объекта перед каждым тестом."""
        self.base_plane = MockBasePlane()
        self.base_plane.scaling_matrix = np.identity(4)
        self.base_plane.translation_matrix = np.identity(4)
        self.extrusion_height = 2.0
        self.polygon = ExtrudedPolygon(self.base_plane, self.extrusion_height)

    def test_init(self):
        """Проверка создания многогранника и корректности углов."""
        self.assertEqual(len(self.polygon.corners), 8)

    def test_update_planes(self):
        """Тест обновления плоскостей."""
        self.polygon.update_planes()
        self.assertEqual(len(self.polygon.planes), 6)

    def test_create_corners(self):
        """Тест создания углов многогранника."""
        self.polygon.create_corners(self.extrusion_height, self.base_plane)
        self.assertEqual(len(self.polygon.corners), 8)

    def test_update_corners(self):
        """Тест обновления углов и плоскостей."""
        with patch.object(self.polygon, "update_planes") as mock_update_planes:
            self.polygon.update_corners()
            self.assertTrue(mock_update_planes.called)

    @patch("src.premitives.Plane.render_self")
    @patch("src.premitives.glColor3f")
    @patch("src.premitives.glBegin")
    @patch("src.premitives.glEnd")
    @patch("src.premitives.glVertex3fv")
    def test_render_self(self, mock_glVertex3fv, mock_glEnd, mock_glBegin, _, __):
        """Тест рендеринга многогранника."""
        self.polygon.render_self()

        self.assertTrue(mock_glBegin.called)
        self.assertTrue(mock_glVertex3fv.called)
        self.assertTrue(mock_glEnd.called)

    def test_pick(self):
        """Тест проверки пересечения луча с многогранником."""
        start = np.array([0, 0, 0])
        direction = np.array([1, 1, 1])
        mat = np.identity(4)

        hit, distance = self.polygon.pick(start, direction, mat)
        self.assertTrue(hit)


class MockBasePlane:
    def __init__(self):
        self.corners = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
            np.array([0, 1, 0]),
        ]


if __name__ == "__main__":
    unittest.main()
