'''

File: tools.py
Author: Hadayat Seddiqi
Date: 10.07.14
Description: A few helpful functions for doing simulated
             quantum annealing simulations.

'''

import numpy as np
import scipy.sparse as sps


def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return [ -1 if k == 1 else 1 for k in vec ]

def spins2bits(vec):
    """ Convert a spinvector @vec to a bitvector. """
    return [ 0 if k == 1 else 1 for k in vec ]

def GenerateNeighbors(nspins, J, maxnb, savepath=None):
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

    Inputs:  @npsins   number of spins in the 2D lattice
             @J        Ising coupling matrix
             @maxnb    the maximum number of neighbors for any spin
                       (if self-connections representing local field
                       terms are present along the diagonal of @J, 
                       this counts as a "neighbor" as well)

    Returns: the above specified "neighbors" list as a numpy array.
    """
    # Precompute neighbors for each spin
    nrows = int(np.sqrt(nspins))
    J = J.todok()  # dictionary of keys type makes this easy
    neighbors = []
    # Iterate over all spins
    for ispin in xrange(nspins):
        nb_pairs = np.zeros((maxnb, 2))
        ipair = 0
        # Find the pairs including this spin
        for pair in J.iterkeys():
            if pair[0] == ispin:
                nb_pairs[ipair] = [ pair[1], J[pair] ]
                ipair += 1
            elif pair[1] == ispin:
                nb_pairs[ipair] = [ pair[0], J[pair] ]
                ipair += 1
        # Record it in the master list
        neighbors.append(nb_pairs)
    J = J.todia()  # DOK is really slow for multiplication
    if savepath is not None:
        np.save(savepath, np.array(neighbors))
    return np.array(neighbors)

def Generate2DIsingInstance(nRows, rng):
    """
    Generate a 2D square Ising model on a torus (with periodic boundaries).
    Couplings are between [-2,2] randomly chosen from a uniform distribution.
    @nRows is the number of rows (and columns) in the 2D lattice.
    
    Returns: Ising matrix in sparse DOK format
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
