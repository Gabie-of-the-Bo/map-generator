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
        p = (h - min_val) / max_val

        return (c2[0]*p + c1[0]*(1 - p), c2[1]*p + c1[1]*(1 - p), c2[2]*p + c1[2]*(1 - p))

    return scale


def weighted_color(cs, ws, min_val, max_val):
    ws = np.divide(ws, np.sum(ws))
    cs = tuple(cs)

    @numba.njit
    def weighted(i, j, h):
        nonlocal ws

        p = (h - min_val) / max_val

        idx = 1
        acc = ws[0]
        pacc = 0

        while acc < p:
            pacc = acc
            acc += ws[idx]
            idx += 1

        amp = max_val - min_val
        min_val2 = min_val + amp * pacc
        max_val2 = min_val + amp * acc
        p = (h - min_val2) / max_val2
        c1 = cs[idx - 1]
        c2 = cs[idx]
        
        return (c2[0]*p + c1[0]*(1 - p), c2[1]*p + c1[1]*(1 - p), c2[2]*p + c1[2]*(1 - p))

    return weighted