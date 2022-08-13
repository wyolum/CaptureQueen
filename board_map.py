import numpy as np
import pylab as pl

def getA(ij):
    i, j = ij.T
    n = len(ij)
    pows = np.array([(i, j) for i in range(3) for j in range(3) if i + j < 4]).T
    A = np.product(ij[:,:,np.newaxis] ** pows[np.newaxis], axis=1)
    #A = np.column_stack([i ** 2, i * j, j ** 2, i, j, np.ones(n)])
    return A

def predict(ij, coeff):
    A = getA(ij)
    return A @ coeff

def fit(ij, xy):
    A = getA(ij)
    out = np.linalg.pinv(A.T @ A) @ (A.T @ xy)
    return out

if __name__ == '__main__':
    dat = np.loadtxt('board_map.txt')
    i = dat[:,1]
    j = dat[:,2]
    x = dat[:,3]
    y = dat[:,4]
    ij = dat[:,1:3]
    xy = dat[:,3:5]
    coeff = fit(ij, xy)
    xy_tilde = predict(ij, coeff)


    fig, ax = pl.subplots(2)
    ax[0].plot(i,x, 'o')
    ax[0].plot(j,x, 'o')
    ax[0].plot(i,xy_tilde[:,0], 'x')
    ax[0].plot(j,xy_tilde[:,0], 'x')

    ax[1].plot(i,y, 'o')
    ax[1].plot(j,y, 'o')
    ax[1].plot(i,xy_tilde[:,1], 'x')
    ax[1].plot(j,xy_tilde[:,1], 'x')
    pl.show()
