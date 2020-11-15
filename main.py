from model.map import Map
from model.layers import *
from algorithms.color import *

layers = {
    0: simple_color((0, 0, 255)),
    1: linear_color((140, 70, 20), (255, 255, 255), 0, 40)
}

world = Map(500, 500).layered(layers, [
        UniformMapLayer(0),
        BlobMapLayer(1, 0).circle(250, 250, 100).rise(1).voronoify(100).voronoify(1000).rough_edges(1).perlin(20).distance_transform(20)
    ])

world.show()