import numpy as np
import numba


def simple_color(color):
    @numba.njit
    def res(i, j, h):
        return color

    return res    


def linear_color(c1, c2, min_val, max_val):
    @numba.njit
    def scale(i, j, h):
        p = (h + min_val) / (min_val + max_val)

        return (c2[0]*p + c1[0]*(1 - p), c2[1]*p + c1[1]*(1 - p), c2[2]*p + c1[2]*(1 - p))

    return scale