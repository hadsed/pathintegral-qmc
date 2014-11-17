'''

File: ising_gen.py
Author: Hadayat Seddiqi
Date: 10.17.14
Description: Generate instances of the 2D lattice Ising model on 
             a torus (i.e. periodic horizontal and vertical 
             boundaries) with coupling strengths between -2 and
             2, no local fields. Output them as numpy ndarrays and
             text files suitable for the "spin-glass server".

'''

import numpy as np
import scipy.sparse as sps


def Generate2DIsing(nRows, rng):
    """
    Generate a 2D square Ising model on a torus (with periodic boundaries).
    Couplings are between [-2,2] randomly chosen from a uniform distribution.
    @nRows is obviously the number of rows in the 2D lattice.
    
    Returns: 4 arrays corresponding to the horizontal, vertical, periodic 
             horizontal, and periodic vertical coupling diagonals in the 
             Ising matrix.
    """

    # Number of rows in 2D square Ising model
    nSpins = nRows**2

    # Generate periodic lattice adjacency matrix
    J = sps.dok_matrix((nSpins,nSpins), dtype=np.float64)
    
    for row in xrange(nSpins):
        # periodic vertical
        if row < nRows:
            J[row, row+(nRows*(nRows-1))] = rng.uniform(low=-2, high=2)
        # loop through columns
        for col in xrange(row, nSpins):
            # periodic horizontal
            if (row % nRows == 0.0):
                J[row, row+nRows-1] = rng.uniform(low=-2, high=2)
            # horizontal
            if (col == row + 1) and (col % nRows != 0.0):
                J[row, col] = rng.uniform(low=-2, high=2)
            # vertical
            if (col == row + nRows):
                J[row, col] = rng.uniform(low=-2, high=2)

    return J.todia().data

    #
    # This doesn't seem to be working right, but should keep it here to fix
    # in the future since it's way faster.
    #

    # # Horizontal nearest-neighbor couplings
    # hcons = rng.uniform(low=-2, high=2, size=nSpins)
    # hcons[::nRows] = 0.0

    # # Vertical nearest-neighbor couplings
    # vcons = rng.uniform(low=-2, high=2, size=nSpins)

    # # Horizontal periodic couplings
    # phcons = np.zeros(nSpins-(nRows-1))
    # phcons[::nRows] = 1

    # phconsIdx = np.where(phcons == 1.0)[0]
    # for i in phconsIdx:
    #     phcons[i] = rng.uniform(low=-2, high=2)
    # # have to pad with zeros because sps.dia_matrix() is too stupid to 
    # # take in diagonal arrays that are the proper length for its offset
    # phcons = np.insert(phcons, 0, [0]*(nRows-1))

    # # Vertical periodic couplings
    # pvcons = rng.uniform(low=-2, high=2, size=nSpins)

    # return hcons, vcons, phcons, pvcons

if __name__ == "__main__":
    # How many instances to generate
    instances = 100
    # Number of rows in the 2D square Ising lattice
    rows = 32
    # Directory to save in, if any (MUST have trailing slash!)
    savedir = 'ising_instances/'

    # Random number generator
    rng = np.random.RandomState()

    for inst in range(instances):
        # Generate a random instance
        h, v, ph, pv = Generate2DIsing(rows, rng)

        # Save as numpy archive
        np.savez(savedir+'inst_'+str(inst), hcons=h, vcons=v, phcons=ph, pvcons=pv, 
                 k=[1, rows, rows-1, 2*rows], nSpins=[rows**2])

        # Now output this to a text file for M. Juenger's spin-glass server
        # These indices start at one, not zero
        hcons_pairs = [ [1+p, 1+p+1, h[p+1]] for p in range(rows**2-1) ]
        vcons_pairs = [ [1+p, 1+p+rows, v[p]] for p in range(rows**2-rows) ]
        phcons_pairs = [ [1+p, 1+p+rows-1, ph[p+2]] for p in range(rows**2-rows+1) ]
        pvcons_pairs = [ [1+p, 1+p+2*rows, pv[p]] for p in range(rows**2-2*rows) ]

        data = np.array(hcons_pairs+vcons_pairs+phcons_pairs+pvcons_pairs)
        data = data[np.all(data != 0.0, axis=1)]  # get rid of zero rows

        np.savetxt(savedir+'inst_'+str(inst)+'.txt', data, fmt='%1d %1d %.12f ')
