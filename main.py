from model.map import Map
from model.layers import *
from algorithms.color import *

layers = {
    0: simple_color((0, 0, 255)),
    1: weighted_color([(244, 164, 96), (96, 128, 56), (56, 102, 0), (140, 70, 20), (255, 255, 255)], [1, 2, 2, 5], 0, 40),
    2: linear_color((150, 150, 150), (70, 70, 70), 1, 16),
    3: simple_color((40, 40, 40))
}

world = Map(500, 500).layered(layers, [
        UniformMapLayer(0),
        BlobMapLayer(1, 0).circle(200, 300, 100).circle(0, 0, 75).rectangle(500, 250, 25, 250).rise(1).voronoify(100).voronoify(1000).rough_edges(1).perlin(20).distance_transform(20),
        BlobMapLayer(2, 1).rectangle(300, 75, 35, 35).voronoify(1000).rough_edges(1).perlin(5).bottom_to(1).distance_transform(5),
        BlobMapLayer(3, 15).maze(300, 75, 15, 15, scale=2).rotate(3, (300, 75))
    ])

world.show((750, 750))