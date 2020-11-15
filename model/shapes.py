import numpy as np

import cv2

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

