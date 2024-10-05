import sys

import numpy
import numpy as np
from src import node

from src.node import Node, ObjectWithControlPoints
from src.premitives import Point, Line, Plane, ExtrudedPolygon, Sphere, Cube, SnowFigure


class Scene:

    # расстояние размещения объекта отонсительно камеры
    PLACE_DEPTH = 5.0

    def __init__(self):
        self.node_list = list()
        self.select_nodes = list()

    def add_node(self, node: Node):
        self.node_list.append(node)

    def render(self):
        for node in self.node_list:
            node.render()

    def apply_for_each_select_nodes(self, function):
        for select_node in self.select_nodes:
            function(select_node)

    def pick(self, start, direction, mat, multiple_choice):
        mindist = sys.maxsize
        closest_node = None
        for node in self.node_list:
            hit, distance = node.pick(start, direction, mat)
            if hit and distance < mindist:
                mindist, closest_node = distance, node

        if closest_node is not None and (
            not self.select_nodes or self.select_nodes and multiple_choice
        ):
            closest_node.select()
            print(f"select: {closest_node}")
            closest_node.depth = mindist
            closest_node.selected_loc = start + direction * mindist
            self.select_nodes.append(closest_node)

            def find_element(primitive):
                return list(
                    filter(
                        lambda select_node: isinstance(select_node, primitive),
                        self.select_nodes,
                    )
                )[0]

            def instance_in_collection(primitive):
                return any(
                    (
                        isinstance(select_node, primitive)
                        for select_node in self.select_nodes
                    )
                )

            if len(self.select_nodes) == 3 and all(
                (isinstance(select_node, Point) for select_node in self.select_nodes)
            ):
                self.create_plane_from_three_points(
                    *[select_node.get_position() for select_node in self.select_nodes]
                )
                for point in self.select_nodes:
                    self.node_list.remove(point)
            elif (
                len(self.select_nodes) == 2
                and instance_in_collection(Point)
                and instance_in_collection(Line)
            ):
                self.create_plane_from_line_and_point(
                    find_element(Line), find_element(Point)
                )
            elif (
                len(self.select_nodes) == 2
                and instance_in_collection(Plane)
                and instance_in_collection(Point)
            ):
                self.create_plane_from_plane_and_point(
                    find_element(Plane), find_element(Point)
                )

        elif self.select_nodes:
            self.apply_for_each_select_nodes(
                lambda select_node: select_node.select(False)
            )
            self.select_nodes.clear()

    def create_plane_from_three_points(self, point1, point2, point3):
        new_plane = Plane.from_three_points(point1, point2, point3)
        self.add_node(new_plane)
        for control_point in new_plane.control_points:
            self.add_node(control_point)

    def create_plane_from_line_and_point(self, line, point):
        self.create_plane_from_three_points(
            node.get_point_coord(line.corners[0], line),
            node.get_point_coord(line.corners[1], line),
            point.get_position(),
        )

    def create_plane_from_plane_and_point(self, plane, point):
        new_point0 = point.get_position() + node.get_point_coord(
            plane.corners[1] - plane.corners[0], plane
        )
        new_point1 = point.get_position() + node.get_point_coord(
            plane.corners[2] - plane.corners[0], plane
        )
        self.create_plane_from_three_points(
            point.get_position(), new_point0, new_point1
        )

    def create_line(self, start, end):
        new_line = Line(start, end)
        self.add_node(new_line)
        for control_point in new_line.control_points:
            self.add_node(control_point)

    def dissection_plane(self):
        if len(self.select_nodes) == 2 and all(
            (isinstance(select_node, Plane) for select_node in self.select_nodes)
        ):
            for point in self.select_nodes[0].control_points:
                self.node_list.remove(point)
            self.select_nodes[0].intersect_with_plane(self.select_nodes[1])
            for point in self.select_nodes[0].control_points:
                self.node_list.append(point)

    def extruded_plane(self):
        if len(self.select_nodes) == 1 and isinstance(self.select_nodes[0], Plane):
            extruded_polygon = ExtrudedPolygon(self.select_nodes[0])
            self.add_node(extruded_polygon)
            for control_point in extruded_polygon.control_points:
                self.add_node(control_point)
            for point in self.select_nodes[0].control_points:
                self.node_list.remove(point)
            self.node_list.remove(self.select_nodes[0])
            self.select_nodes.clear()

    def scale_selected(self, up):
        if not self.select_nodes:
            return
        self.apply_for_each_select_nodes(lambda select_node: select_node.scale(up))

    def rotate_selected_color(self, forwards):
        if not self.select_nodes:
            return
        self.apply_for_each_select_nodes(
            lambda select_node: select_node.rotate_color(forwards)
        )

    def delete_selected(self):
        for select_node in self.select_nodes:
            if isinstance(select_node, ObjectWithControlPoints):
                for point in select_node.control_points:
                    self.node_list.remove(point)
            self.node_list.remove(select_node)
        self.select_nodes.clear()

    def move_selected(self, start, direction, inv_modelview):
        """Двигает выделенный узел"""
        if not self.select_nodes:
            return

        def move_each_node(select_node):
            node = select_node
            depth = node.depth
            oldloc = node.selected_loc

            newloc = start + direction * depth

            translation = newloc - oldloc
            pre_tran = numpy.array([translation[0], translation[1], translation[2], 0])
            translation = inv_modelview.dot(pre_tran)

            node.translate(translation[0], translation[1], translation[2])
            node.selected_loc = newloc

        self.apply_for_each_select_nodes(move_each_node)

    def combine(self):
        if not self.select_nodes:
            return None

        new_node = node.HierarchicalNode()
        new_node.child_nodes = self.select_nodes[:]

        # Обновляем AABB для нового узла, используя все дочерние узлы
        min_corner = np.min(
            [child.get_transformed_aabb().min_point for child in new_node.child_nodes],
            axis=0,
        )
        max_corner = np.max(
            [child.get_transformed_aabb().max_point for child in new_node.child_nodes],
            axis=0,
        )
        new_node.aabb = node.AABB(min_corner, max_corner)

        # Добавляем новый узел в сцену и очищаем список выделенных узлов
        self.node_list.append(new_node)
        self.delete_selected()

        for child_node in new_node.child_nodes:
            child_node.color_index = len(self.node_list)

        return new_node

    def place(self, shape, start, direction, inv_modelview):
        """Размещает новую плоскость"""
        new_node = None
        if shape == "sphere":
            new_node = Sphere()
        elif shape == "cube":
            new_node = Cube()
        elif shape == "figure":
            new_node = SnowFigure()
        elif shape == "point":
            new_node = Point()

        self.add_node(new_node)

        translation = start + direction * self.PLACE_DEPTH

        pre_tran = numpy.array([translation[0], translation[1], translation[2], 1])
        translation = inv_modelview.dot(pre_tran)

        new_node.translate(translation[0], translation[1], translation[2])
        print(f"new node: {str(new_node)}")
