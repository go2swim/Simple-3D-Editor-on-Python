import json
import os
from datetime import datetime
from json import JSONEncoder

import numpy as np
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_FRONT, glReadBuffer

from src.node import HierarchicalNode, ObjectWithControlPoints
from src.premitives import (
    ActivePoint,
    Cube,
    Sphere,
    SnowFigure,
    Line,
    Point,
    Plane,
    ExtrudedPolygon,
)
from src.scene import Scene
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
from PIL import Image
from OpenGL.GLUT import glutGet, GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT

SAVE_DIRECTORY = (
    "./data/Save_scene"
    if os.path.basename(os.getcwd()) == "3d_editor"
    else "../data/Save_scene"
)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_name_file_for_save_scene():
    return f'scene_{str(datetime.now()).replace(":", "-").split(".")[0]}'


def save_scene(scene):
    scene_data = {
        "nodes": [
            node.to_dict()
            for node in scene.node_list
            if not isinstance(node, ActivePoint)
        ]
    }

    name_with_date = get_name_file_for_save_scene()

    with open(f"./{SAVE_DIRECTORY}/{name_with_date}.json", "w") as file:
        json.dump(scene_data, file, indent=4, cls=NumpyArrayEncoder)  # отступ от :
    print("Scene saved")


def load_scene(filename):
    with open(f"{SAVE_DIRECTORY}/{filename}", "r") as file:
        scene_data = json.load(file)
    return load_data(scene_data)


def load_data(scene_data):
    scene = Scene()

    for node_data in scene_data["nodes"]:
        node_type = node_data["type"]
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
            node.child_nodes = node_data.get("children", [])

        if node:
            node.color_index = node_data.get("color_index", 0)
            position = node_data.get("position", [0, 0, 0])
            node.translate(*position)

            if isinstance(node, ObjectWithControlPoints):
                node.corners = np.array(node_data.get("corners"))
                node.create_control_points()
                for control_point in node.control_points:
                    scene.add_node(control_point)

                if isinstance(node, ExtrudedPolygon):
                    node.update_planes()
                elif isinstance(node, Line):
                    node.update_aabb()

            scene.add_node(node)

    print("Scene is loaded")
    return scene


def get_saved_scenes():
    return [
        f
        for f in os.listdir(SAVE_DIRECTORY)
        if os.path.isfile(os.path.join(SAVE_DIRECTORY, f))
    ]


def export_scene_to_image():
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)

    glReadBuffer(GL_FRONT)  # Читаем с переднего буфера кадра
    pixel_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    image = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 3)

    # Переворачиваем изображение по вертикали (так как OpenGL хранит его снизу вверх)
    image = np.flipud(image)

    filename = get_name_file_for_save_scene() + ".png"
    path = SAVE_DIRECTORY.replace(
        os.path.basename(SAVE_DIRECTORY), "Save_scene_as_image"
    )

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, filename)

    img = Image.fromarray(image)
    img.save(path)
    print(f"Scene in image format saved as {filename}")
