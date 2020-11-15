import numba
import numpy as np

from model.layers import *

class Map:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.composed = False

    # Functional generation interface
    def layered(self, layerDescription, layers=[]):
        self.layerDescription = layerDescription
        self.layers = layers

        return self


    def compose(self):
        self.mask = np.zeros((self.height, self.width), bool)
        self.heightMap = np.zeros((self.height, self.width), np.int32)
        self.idMap = np.zeros((self.height, self.width), np.int32)
        self.color = np.zeros((self.height, self.width, 3), np.uint8)

        @numba.njit(parallel=True)
        def update_color(func, mask: np.ndarray, height: np.ndarray, dest: np.ndarray):
            h, w = mask.shape

            for i in numba.prange(h):
                for j in numba.prange(w):
                    if mask[i, j]:
                        dest[i, j] = func(i, j, height[i, j])

        if not self.composed:
            for i, l in enumerate(self.layers):
                l.compose(self)
                
                self.mask |= l.mask

                height_mask = (l.heightMap >= self.heightMap) & l.mask

                self.heightMap[height_mask] = l.heightMap[height_mask]
                self.idMap[height_mask] = l.id
                update_color(self.layerDescription[l.id], height_mask, l.heightMap, self.color)

            self.composed = True

        return self
   
    def show(self):
        if not self.composed:
            self.compose()

        cv.imshow("Map visualization", cv.cvtColor(self.color, cv.COLOR_RGB2BGR))
        cv.waitKey()