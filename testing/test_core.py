'''

File: test_tools.py
Author: Hadayat Seddiqi
Date: 01.16.15
Description: Test some of the tools included in piqmc.

'''

import numpy as np
import scipy.sparse as sps

import piqmc.tools as tools


def test_spinbitconversion():
    a = np.array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.])
    b = tools.bits2spins(a)
    c = tools.spins2bits(b)
    assert(np.all(a == [0,1,0,1,0,1,0,1,0,1]))
    assert(np.all(b == [1,-1,1,-1,1,-1,1,-1,1,-1]))
    assert(np.all(c == a))

def test_generateising():
    nrows = 4
    rng = np.random.RandomState(123)
    J = tools.Generate2DIsingInstance(nrows, rng)
    assert((J - sps.triu(J)).nnz == 0)

def test_generateneighbors():
    J = sps.dok_matrix((5,5), dtype=np.float64)
    #
    #  We're going to build this graph:
    #
    #   0 --- 1
    #   |   / |
    #   |  4  |
    #   | /   |
    #   3 --- 2
    #
    J[0,1] = 1
    J[1,2] = 1
    J[2,3] = 1
    J[3,4] = 1
    J[0,3] = 1
    J[1,4] = 1
    J[0,4] = 1
    J[0,0] = 1
    J[1,1] = 1
    J[2,2] = 1
    J[3,3] = 1
    J[4,4] = 1
    true_nb = np.array(
        [[[ 1.,  1.],
          [ 0.,  1.],
          [ 4.,  1.],
          [ 3.,  1.]],

         [[ 0.,  1.],
          [ 2.,  1.],
          [ 4.,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 2.,  1.],
          [ 3.,  1.],
          [ 0.,  0.]],

         [[ 3.,  1.],
          [ 2.,  1.],
          [ 0.,  1.],
          [ 4.,  1.]],

         [[ 4.,  1.],
          [ 1.,  1.],
          [ 0.,  1.],
          [ 3.,  1.]]]
        )
    generated_nb = tools.GenerateNeighbors(nspins=5, J=J, maxnb=4,
                                           savepath=None)
    assert(np.all(true_nb == generated_nb))
