import numpy as np
import cv2 as cv
import numba

import sobol_seq as sobol
from perlin_numpy import generate_fractal_noise_2d
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

from model.shapes import *

class UniformMapLayer:
    """Map layer defined solely by a height. Can be used to build bases (water, for example)"""

    def __init__(self, id, height=0):
        self.id = id
        self.height = height

        self.composed = False
    
    # Lazy generation
    def compose(self, wmap):
        h, w = wmap.height, wmap.width

        if not self.composed:
            self.heightMap = np.ones((h, w), np.int32) * self.height
            self.mask = np.ones((h, w), bool)

            self.composed = True


class BlobMapLayer:
    """Map layer defined by a series of shapes and subsequent transformations"""

    def __init__(self, id, height=0):
        self.id = id
        self.height = height
        self.shapes = []
        self.transformations = []

        self.composed = False    
    
    # Shapes
    def circle(self, i, j, radius):
        self.shapes.append((i, j, Circle(radius)))

        return self

    # Transformations
    def perlin(self, scale: int):
        def func(mask: np.ndarray, heightMap: np.ndarray):
            scaleY = 128 * max(int(mask.shape[0] / 128), 1)
            scaleX = 128 * max(int(mask.shape[1] / 128), 1)

            pnoise = generate_fractal_noise_2d((scaleY, scaleX), (8, 8), 5)
            pnoise = cv.resize(pnoise, mask.shape)

            pnoise *= mask
            pnoise *= scale
            
            heightMap[:, :] = (heightMap + pnoise).astype(np.int32)

        self.transformations.append(func)

        return self


    def distance_transform(self, scale: int):
        def func(mask: np.ndarray, heightMap: np.ndarray):
            h = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 3)
            cv.normalize(h, h, 1, 0, cv.NORM_INF)
            
            heightMap += (h * scale).astype(np.uint32)

        self.transformations.append(func)

        return self


    def voronoify(self, points: int):
        def func(mask: np.ndarray, heightMap: np.ndarray):
            rnd = sobol.i4_sobol_generate(2, points)
            rnd[:, 0] *= mask.shape[0]
            rnd[:, 1] *= mask.shape[1]

            vor = Voronoi(rnd)

            colorMask = mask.astype(np.uint8)

            for region in vor.regions:
                region = np.array([vor.vertices[i] for i in region if i != -1]).astype(np.int32)

                if region.shape[0] > 2:
                    cpy = np.zeros_like(colorMask)
                    cv.fillPoly(cpy, np.array([region], dtype=np.int32), 1)

                    a = np.sum(cpy)
                    b = np.sum(cpy & colorMask)

                    if b / a > 0:
                        mask[cpy.astype(bool)] = True


        self.transformations.append(func)

        return self


    def rough_edges(self, coef: float):
        def func(mask: np.ndarray, heightMap: np.ndarray):
            my, mx = np.indices(mask.shape, dtype=np.float32)
            mx += cv.GaussianBlur(np.random.normal(0, coef, mask.shape), (3, 3), 0)
            my += cv.GaussianBlur(np.random.normal(0, coef, mask.shape), (3, 3), 0)

            mx, my = cv.convertMaps(mx, my, cv.CV_16SC2)

            mask[:, :] = cv.remap(mask.astype(np.uint8), mx, my, cv.INTER_NEAREST).astype(bool)
            cv.remap(heightMap, mx, my, cv.INTER_NEAREST, heightMap)

        self.transformations.append(func)

        return self


    def rise(self, height: int):
        def func(mask: np.ndarray, heightMap: np.ndarray):
            heightMap += height

        self.transformations.append(func)

        return self


    def sink(self, height: int):
        def func(mask: np.ndarray, heightMap: np.ndarray):
            heightMap -= height

        self.transformations.append(func)

        return self

    # Lazy generation
    def compose(self, wmap):
        h, w = wmap.height, wmap.width

        @numba.njit(parallel=True)
        def update_shape(i: int, j: int, shape: np.ndarray, dest: np.ndarray):
            sh, sw = shape.shape

            for ii in numba.prange(sh):
                for jj in numba.prange(sw):
                    di = ii - sh // 2 + i
                    dj = jj - sw // 2 + j

                    if di >= 0 and di < dest.shape[0] and dj >= 0 and dj < dest.shape[1]:
                        dest[di, dj] |= shape[ii, jj]

        if not self.composed:
            self.mask = np.zeros((h, w), bool)
            self.heightMap = np.zeros((h, w), np.int32)

            # Draw shapes
            for i, j, s in self.shapes:
                sm = s.as_matrix()
                update_shape(i, j, sm, self.mask)

            self.heightMap[self.mask] = self.height

            # Transformations
            for t in self.transformations:
                t(self.mask, self.heightMap)

            self.composed = True