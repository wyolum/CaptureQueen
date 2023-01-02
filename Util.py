import numpy as np
import pylab as pl

def linfit(x, y):
    A = np.column_stack([np.ones_like(x), x])
    coeff = np.linalg.pinv(A) @ y
    return coeff

def test_linfit():
    x = np.arange(10)
    a0 = 1.234
    a1 = 5.678
    y = a1 * x + a0

    coeff = linfit(x, y)
    assert np.abs(coeff[0] - a0) < 1e-8
    assert np.abs(coeff[1] - a1) < 1e-8

test_linfit()

def intersect(l1, l2):
    b, a = l1
    d, c = l2
    x = (d - b) / (a - c)
    y1 = a * x + b
    y2 = c * x + d
    y = (y1 + y2) / 2
    return np.array([x, y])

def flatmeshgrid(*args):
    _out = np.meshgrid(*args)
    return tuple(o.ravel() for o in _out)

def getA(i, j):
    A = np.column_stack([i ** 2, i * j, j ** 2, i, j, np.ones_like(i)])
    return A

def plane(corners):
    ni, nj = corners.shape[:2]
    _i = np.arange(ni)
    _j = np.arange(nj)
    j, i = flatmeshgrid(_j, _i)
    A = getA(i, j)
    cx = np.linalg.pinv(A) @ corners[:,:,0].ravel()
    cy = np.linalg.pinv(A) @ corners[:,:,1].ravel()
    return cx, cy

def eval_plane(cx, cy, i, j):
    A = getA(i, j)
    x = A @ cx
    y = A @ cy
    return np.column_stack([x, y])

if __name__ == '__main__':
    corners = np.array([[[0, 0], [0, 1]],
                        [[1, 0], [1, 1]]])

    DEG = np.pi / 180
    theta = 20 * DEG
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    corners = corners @ R


    cx, cy = plane(corners)
    _i = np.arange(8)
    _j = np.arange(8)
    i, j = flatmeshgrid(_i, _j)
    xy = eval_plane(cx, cy, i, j)

    pl.figure()
    pl.plot(corners[:,:,0].ravel(),corners[:,:,1].ravel(), 'bo')

    pl.plot(xy[:,0], xy[:,1], 'r.')
    pl.show()
    input('here')
