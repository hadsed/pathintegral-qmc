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


if __name__ == "__main__":
    # How many instances to generate
    instances = 1
    # Number of rows in the 2D square Ising lattice
    rows = 32
    # Directory to save in, if any (MUST have trailing slash!)
    savedir = 'ising_instances/'
    # Random number generator
    rng = np.random.RandomState()
    # Generate them
    for inst in range(instances):
        # Generate a random instance
        isingJ = Generate2DIsing(rows, rng)
        # Save as numpy archive
        h, v, ph, pv = isingJ.todia().data
        np.savez(savedir+'inst_'+str(inst)+'_'+str(rows)+'x'+str(rows),
                 hcons=h, vcons=v, phcons=ph, pvcons=pv, 
                 k=[1, rows, rows-1, 2*rows], nSpins=[rows**2])
        # Now output this to a text file for M. Juenger's spin-glass server
        data = [ (i+1,j+1,k) for ((i,j), k) in isingJ.items() ]
        np.savetxt(savedir+'inst_'+str(inst)+'_'+str(rows)+'x'+str(rows)+'.txt', 
                   data, fmt='%1d %1d %.12f ')
