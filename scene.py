import sys

import numpy
import numpy as np

from node import Node
import node


class Scene:

    #растояние размещения объекта отонсительно камеры
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
        """ Выделяет узел """

        # находим ближайший узел
        mindist = sys.maxsize
        closest_node = None
        for node in self.node_list:
            hit, distance = node.pick(start, direction, mat)
            if hit and distance < mindist:
                mindist, closest_node = distance, node

        # Отмечаем, если что-то нашли
        if closest_node is not None and (not self.select_nodes or self.select_nodes and multiple_choice):
            closest_node.select()
            closest_node.depth = mindist
            closest_node.selected_loc = start + direction * mindist
            self.select_nodes.append(closest_node)
        elif self.select_nodes:
            self.apply_for_each_select_nodes(lambda select_node: select_node.select(False))
            self.select_nodes.clear()

    def scale_selected(self, up):
        if not self.select_nodes:
            return
        self.apply_for_each_select_nodes(lambda select_node: select_node.scale(up))

    def rotate_selected_color(self, forwards):
        """ Rotate the color of the currently selected node """
        if not self.select_nodes:
            return
        self.apply_for_each_select_nodes(lambda select_node: select_node.rotate_color(forwards))

    def delete_selected(self):
        for select_node in self.select_nodes:
            self.node_list.remove(select_node)
        self.select_nodes.clear()

    def move_selected(self, start, direction, inv_modelview):
        """ Двигает выделенный узел"""
        if not self.select_nodes: return
        def move_each_node(select_node):
            # Find the current depth and location of the selected node
            node = select_node
            depth = node.depth
            oldloc = node.selected_loc

            # The new location of the node is the same depth along the new ray
            newloc = (start + direction * depth)

            # transform the translation with the modelview matrix
            translation = newloc - oldloc
            pre_tran = numpy.array([translation[0], translation[1], translation[2], 0])
            translation = inv_modelview.dot(pre_tran)

            # translate the node and track its location
            node.translate(translation[0], translation[1], translation[2])
            node.selected_loc = newloc

        self.apply_for_each_select_nodes(move_each_node)

    def combine(self):
        if not self.select_nodes:
            return None

        new_node = node.HierarchicalNode()
        new_node.child_nodes = self.select_nodes[:]

        # Обновляем AABB для нового узла, используя все дочерние узлы
        min_corner = np.min([child.get_transformed_aabb().min_point for child in new_node.child_nodes], axis=0)
        max_corner = np.max([child.get_transformed_aabb().max_point for child in new_node.child_nodes], axis=0)
        new_node.aabb = node.AABB(min_corner, max_corner)

        # Добавляем новый узел в сцену и очищаем список выделенных узлов
        self.node_list.append(new_node)
        self.delete_selected()

        for child_node in new_node.child_nodes:
            child_node.color_index = len(self.node_list)

        return new_node

    def place(self, shape, start, direction, inv_modelview):
        """ Размещает новую плоскость """
        new_node = None
        if shape == 'sphere':
            new_node = node.Sphere()
        elif shape == 'cube':
            new_node = node.Cube()
        elif shape == 'figure':
            new_node = node.SnowFigure()
        elif shape == 'point':
            new_node = node.Point()

        self.add_node(new_node)

        # place the node at the cursor in camera-space
        translation = (start + direction * self.PLACE_DEPTH)

        # convert the translation to world-space
        pre_tran = numpy.array([translation[0], translation[1], translation[2], 1])
        translation = inv_modelview.dot(pre_tran)

        new_node.translate(translation[0], translation[1], translation[2])
        print(f'new node: {str(new_node)}')