import json
import os
from datetime import datetime
from json import JSONEncoder

import numpy as np

from node import HierarchicalNode, ObjectWithControlPoints
from premitives import ActivePoint, Cube, Sphere, SnowFigure, Line, Point, Plane, ExtrudedPolygon
from scene import Scene

SAVE_DIRECTORY = './Save_scene'


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def save_scene(scene):
    # Преобразуем объекты сцены в сериализуемый формат
    scene_data = {
        "nodes": [node.to_dict() for node in scene.node_list if not isinstance(node, ActivePoint)]
    }

    name_with_date = f'scene_{str(datetime.now()).replace(':', '-').split('.')[0]}'

    # Сохраняем данные в файл
    with open(f"./{SAVE_DIRECTORY}/{name_with_date}.json", "w") as file:
        json.dump(scene_data, file, indent=4, cls=NumpyArrayEncoder)  # отступ от :
    print("Scene saved")


def load_scene(filename):
    with open(f'{SAVE_DIRECTORY}/{filename}', "r") as file:
        scene_data = json.load(file)
    return load_data(scene_data)


def load_data(scene_data):
    scene = Scene()  # Создаем новую сцену

    # Проходим по всем узлам в сохраненной сцене
    for node_data in scene_data['nodes']:
        node_type = node_data['type']
        node = None

        if node_type == "Cube":
            node = Cube()

        elif node_type == "Sphere":
            node = Sphere()

        elif node_type == "SnowFigure":
            node = SnowFigure()

        elif node_type == "Line":
            node = Line([0, 0, 0], [1, 1, 1])

        elif node_type == "Point":
            node = Point()

        elif node_type == "Plane":
            node = Plane()

        elif node_type == "ExtrudedPolygon":
            node = ExtrudedPolygon(Plane())

        elif node_type == "HierarchicalNode":
            node = HierarchicalNode()
            node.child_nodes = node_data.get('children', [])

        # Восстанавливаем общее состояние узла (цвет, позиция)
        if node:
            node.color_index = node_data.get('color_index', 0)
            position = node_data.get('position', [0, 0, 0])
            node.translate(*position)

            if isinstance(node, ObjectWithControlPoints):
                node.corners = np.array(node_data.get('corners'))
                node.create_control_points()
                for control_point in node.control_points:
                    scene.add_node(control_point)

                if isinstance(node, ExtrudedPolygon):
                    node.update_planes()
                elif isinstance(node, Line):
                    node.update_aabb()

            scene.add_node(node)

    print('Scene is loaded')
    return scene


def get_saved_scenes():
    return [f for f in os.listdir(SAVE_DIRECTORY) if os.path.isfile(os.path.join(SAVE_DIRECTORY, f))]
