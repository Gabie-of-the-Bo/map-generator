import numpy as np
import numba
import cv2 as cv

import random

class Circle:
    def __init__(self, radius):
        self.radius = radius

    # Matrix representation
    def as_matrix(self):
        r = self.radius * 2 + 1
        m = r // 2

        y, x = np.indices((r, r))
        mod = np.sqrt((y - m)**2 + (x - m)**2)

        return mod <= self.radius
    
class Rectangle:
    def __init__(self, ampX, ampY):
        self.ampX = ampX
        self.ampY = ampY

    # Matrix representation
    def as_matrix(self):
        return np.ones((self.ampY * 2 + 1, self.ampX * 2 + 1), np.bool)

    
class Maze:
    def __init__(self, ampX, ampY, scale=1):
        self.ampX = ampX
        self.ampY = ampY
        self.scale = scale

    # Matrix representation
    def as_matrix(self):
        res = np.ones((self.ampY * 2 + 1, self.ampX * 2 + 1), np.bool)

        @numba.njit
        def dft(maze: np.ndarray, start):
            stack = [start]
            visited = set([(0, 0)])
            pred = {start: start}

            idx = np.array([0, 1, 2, 3])
            increments = ((0, 2), (2, 0), (0, -2), (-2, 0))

            while stack:
                v = stack.pop()

                if v in visited:
                    continue

                i, j = v
                ip, jp = pred[v]

                maze[i, j] = False
                maze[(i + ip) // 2, (j + jp) // 2] = False

                visited.add(v)

                adj = [(i + ii, j + jj) for ii, jj in increments]

                np.random.shuffle(idx)

                for v2_idx in idx:
                    v2 = adj[v2_idx]
                    i2, j2 = v2

                    if i2 >= 0 and i2 < maze.shape[0] and j2 >= 0 and j2 < maze.shape[1] - 1:
                        if v2 in visited:
                            pass

                        else:
                            stack.append(v2)
                            pred[v2] = v

        dft(res, (1, 1))

        if self.scale > 1:
            res = cv.resize(res.astype(np.uint8), (int(res.shape[0] * self.scale), int(res.shape[1] * self.scale)), interpolation=cv.INTER_NEAREST).astype(np.bool)

        return res