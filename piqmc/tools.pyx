# encoding: utf-8
# cython: profile=False
# filename: tools.pyx
'''

File: tools.py
Author: Hadayat Seddiqi
Date: 10.07.14
Description: A few helpful functions for doing simulated
             and quantum annealing simulations.

'''

import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sps


def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return [ -1 if k == 1 else 1 for k in vec ]

def spins2bits(vec):
    """ Convert a spinvector @vec to a bitvector. """
    return [ 0 if k == 1 else 1 for k in vec ]

cpdef GenerateNeighbors(int nspins, 
                        J,  # scipy.sparse matrix
                        int maxnb, 
                        str savepath=None):
    """
    Precompute a list that include neighboring indices to each spin
    and the corresponding coupling value. Specifically, build:

    neighbors = [
           [ [ ni_0, J[0, ni_0] ], 
             [ ni_1, J[0, ni_1] ], 
               ... ],

           [ [ ni_0, J[1, ni_0] ], 
             [ ni_1, J[1, ni_1] ], 
               ... ],

            ...

           [ [ ni_0, J[nspins-1, ni_0]], 
             [ ni_1, J[nspins-1, ni_1]],                   
               ... ]
     ]

    For graphs that are not completely "regular", there will be
    some rows in the neighbor matrix for each spin that will show
    [0,0]. This is required to keep the neighbors data structure
    an N-dimensional array, but in the energy calculations will have
    no contribution as the coupling strength is essentially zero.
    On the other hand, this is why @maxnb must be set to as high a
    number as necessary, but no more (otherwise it will incur some
    computational cost).

    Args:
        @nspins (np.array, float): number of spins in the 2D lattice
        @J (sp.sparse.matrix, float): Ising coupling matrix
        @maxnb (int): the maximum number of neighbors for any spin
                  (if self-connections representing local field
                  terms are present along the diagonal of @J, 
                  this counts as a "neighbor" as well)

    Returns:
        np.ndarray, float:  the above specified "neighbors" list 
                            as a 3D numpy array
    """
    # predefining vars
    cdef int ispin = 0
    cdef int ipair = 0
    # the neighbors data structure
    cdef np.float_t[:, :, :]  nbs = np.zeros((nspins, maxnb, 2))
    # dictionary of keys type makes this easy
    J = J.todok()
    # Iterate over all spins
    for ispin in xrange(nspins):
        ipair = 0
        # Find the pairs including this spin
        for pair in J.iterkeys():
            if pair[0] == ispin:
                nbs[ispin, ipair, 0] = pair[1]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
            elif pair[1] == ispin:
                nbs[ispin, ipair, 0] = pair[0]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
    J = J.tocsr()  # DOK is really slow for multiplication
    if savepath is not None:
        np.save(savepath, nbs)
    return nbs

def Generate2DIsingInstance(nRows, rng):
    """
    Generate a 2D square Ising model on a torus (with periodic boundaries).
    Couplings are between [-2,2] randomly chosen from a uniform distribution.
    @nRows is the number of rows (and columns) in the 2D lattice.
    
    Args:
        @nRows (int): number of rows in the square Ising matrix
        @rng (np.RandomState): numpy random number generator object

    Returns:
        sp.sparse.dok_matrix, float: Ising matrix in sparse DOK format
    """
    # Number of rows in 2D square Ising model
    nSpins = nRows**2
    # Generate periodic lattice adjacency matrix
    J = sps.dok_matrix((nSpins,nSpins), dtype=np.float64)
    for row in xrange(nSpins):
        # periodic vertical (consider first "row" in square lattice only)
        if row < nRows:
            J[row, row+(nRows*(nRows-1))] = rng.uniform(low=-2, high=2)
        # loop through columns
        for col in xrange(row, nSpins):
            # periodic horizontal
            if (row % nRows == 0.0):
                J[row, row+nRows-1] = rng.uniform(low=-2, high=2)
            # horizontal neighbors (we can build it all using right neighbors)
            if (col == row + 1) and (row % nRows != nRows - 1):  # right neighbor
                J[row, col] = rng.uniform(low=-2, high=2)
            # vertical neighbors (we can build it all using bottom neighbors)
            if (col == row + nRows):
                J[row, col] = rng.uniform(low=-2, high=2)
    return J

def Generate2DLattice(int nrows, 
                      int ncols, 
                      rng, 
                      int periodic=0):
    """
    Generate a 2D square Ising model on a torus (with periodic boundaries).
    Couplings are between [-1e-8,1e-8] randomly chosen from a uniform distribution.
    
    Note: in the code there is mixed usage of "rows" and "columns". @nrows
    talks about the number of rows in the lattice, but the jrow variable
    references the row index of the J matrix (i.e. a particular spin).

    Args:
        @nrows (int): number of rows in the 2D Ising matrix
        @ncols (int): number of columns in the 2D Ising matrix
        @rng (np.RandomState): numpy random number generator object
        @periodic (int): binary number specifying periodic boundary
                         conditions (1) or not (0)

    Returns:
        sp.sparse.dok_matrix, float: Ising matrix in sparse DOK format
    """
    cdef int nspins = nrows*ncols
    cdef int jrow = 0
    cdef int jcol = 0
    # Generate periodic lattice adjacency matrix
    J = sps.dok_matrix((nspins,nspins), dtype=np.float64)
    for jrow in xrange(nspins):
        # periodic vertical (consider first "row" in square lattice only)
        if (jrow < ncols) and periodic:
            J[jrow, jrow + ncols*(nrows-1)] = rng.uniform(low=-1e-8, high=1e-8)
        # periodic horizontal
        if (jrow % ncols == 0.0) and periodic:
            J[jrow, jrow+ncols-1] = rng.uniform(low=-1e-8, high=1e-8)
        # loop through columns
        for jcol in xrange(jrow, nspins):
            # horizontal neighbors (we can build it all using right neighbors)
            if ((jcol == jrow + 1) and 
                (jrow % ncols != ncols - 1)):  # right neighbor
                J[jrow, jcol] = rng.uniform(low=-1e-8, high=1e-8)
            # vertical neighbors (we can build it all using bottom neighbors)
            if (jcol == jrow + ncols):
                J[jrow, jcol] = rng.uniform(low=-1e-8, high=1e-8)
    return J

def GenerateKblockLattice(int nrows, 
                          int ncols, 
                          rng, 
                          int k=1):
    """
    Generate an Ising model that extends the 2D lattice case to where each
    spin has coupling to all other spins in a "block" of radius @k.

    @k = 1:
             o------o------o
             |      |      |
             |      |      |
             o-----|Z|-----o
             |      |      |
             |      |      |
             o------o------o

    @k = 2:

      o------o------o------o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o------o------o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o-----|Z|-----o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o------o------o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o------o------o------o


    where each 'o' is directly coupled to 'Z', the central spin we're 
    considering. This forms a kind of receptive field around each neuron.
    Couplings are between [-1e-8,1e-8] randomly chosen from a uniform distribution.
    
    Args:
        @nrows (int): number of rows in the 2D Ising matrix
        @ncols (int): number of columns in the 2D Ising matrix
        @rng (np.RandomState): numpy random number generator object
        @k (int): block size as explained above

    Returns:
        sp.sparse.dok_matrix, float: Ising matrix in sparse DOK format
    """
    cdef int nspins = nrows*ncols
    cdef int ispin = 0
    cdef int jspin = 0
    cdef int ki = 0
    cdef int tlc = 0
    cdef int indicator = 0

    # Generate periodic lattice adjacency matrix
    J = sps.dok_matrix((nspins,nspins), dtype=np.float64)
    # loop through spins
    for ispin in xrange(nspins):
        # loop over the radii (no self-connections, so no zero)
        for ki in xrange(1,k+1):
            # put bounds on what the true radius is in each direction
            # so it doesn't go beyond the boundaries
            kleft = ispin % ncols
            kleft = ki if kleft >= ki else kleft
            if ((ispin+1) % ncols) != 0:
                kright = ncols - ((ispin+1) % ncols)
            else:
                kright = 0
            kright = ki if kright >= ki else kright
            kup = ki
            while ispin - kup*ncols < 0:
                kup -= 1
            kup = ki if kup >= ki else kup
            kdown = ki
            while ispin + kdown*ncols >= nspins:
                kdown -= 1
            kdown = ki if kdown >= ki else kdown
            # top left corner spin (make sure it's not a negative index)
            tlc = max(ispin - (kup)*ncols - kleft, 0)
            # loop over all spins at ki-th radius
            # upper side
            for idx in (tlc+r for r in xrange(kleft+kright+1)):
                if ispin < idx:
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # lower side
            for idx in (tlc+r+(kup+kdown)*ncols for r in xrange(kleft+kright+1)):
                if ispin < idx:
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # left side
            for idx in (tlc+r*ncols for r in xrange(1,kup+kdown+1)):
                if ispin < idx:
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # right side
            for idx in (tlc+kleft+kright+r*ncols for r in xrange(1,kup+kdown+1)):
                if ispin < idx:
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
    return J
