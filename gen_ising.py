'''

File: gen_ising.py
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


if __name__ == "__main__":
    # Get some command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", 
                        default=16,
                        nargs='?',
                        type=int,
                        help="Number of rows in square 2D Ising lattice.")
    parser.add_argument("--instances", 
                        default=1,
                        nargs='?',
                        type=int,
                        help="Number of instances to generate.")
    parser.add_argument("--savedir", 
                        default='ising_instances/',
                        nargs='?',
                        type=str,
                        help="Save in this directory.")
    # Parse the inputs
    args = parser.parse_args()
    # How many instances to generate
    instances = args.inst
    # Number of rows in the 2D square Ising lattice
    rows = args.rows
    # Directory to save in, if any
    savedir = args.savedir + '/'
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
