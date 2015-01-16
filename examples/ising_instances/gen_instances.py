'''

File: gen_instances.py
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


#
# Generate a 2D square Ising model on a torus (with periodic boundaries)
#

inst = 0

# Random number generator
rng = np.random.RandomState()

# Number of rows in 2D square Ising model
nRows = 3
nSpins = nRows**2

# Horizontal nearest-neighbor couplings
hcons = rng.uniform(low=-2, high=2, size=nSpins)
hcons[::nRows] = 0.0
print hcons
# Vertical nearest-neighbor couplings
vcons = rng.uniform(low=-2, high=2, size=nSpins)

# Horizontal periodic couplings
phcons = np.zeros(nSpins-(nRows-1))
phcons[::nRows] = 1

phconsIdx = np.where(phcons == 1.0)[0]
for i in phconsIdx:
    phcons[i] = rng.uniform(low=-2, high=2)
# have to pad with zeros because sps.dia_matrix() is too stupid to 
# take in diagonal arrays that are the proper length for its offset
phcons = np.insert(phcons, 0, [0,0])

# Vertical periodic couplings
pvcons = rng.uniform(low=-2, high=2, size=nSpins)

# Construct the sparse diagonal matrix
isingJ = sps.dia_matrix(([hcons, vcons, phcons, pvcons],
                         [1, nRows, nRows-1, 2*nRows]),
                        shape=(nSpins, nSpins))

np.savez('inst_'+str(inst), hcons=hcons, vcons=vcons, phcons=phcons, 
         pvcons=pvcons, k=[1, nRows, nRows-1, 2*nRows], nSpins=[nSpins])

loader = np.load('inst_'+str(inst)+'.npz')

isingJ_check = sps.dia_matrix(([loader['hcons'], loader['vcons'],
                          loader['phcons'], loader['pvcons']],
                         loader['k']),
                        shape=(loader['nSpins'][0], loader['nSpins'][0]))

# print isingJ == isingJ_check

# Now output this to a text file for M. Juenger's spin-glass server
# These indices start at one, not zero
hcons_pairs = [ [1+p, 1+p+1, hcons[p+1]] for p in range(nSpins-1) ]
vcons_pairs = [ [1+p, 1+p+nRows, vcons[p]] for p in range(nSpins-nRows) ]
phcons_pairs = [ [1+p, 1+p+nRows-1, phcons[p+2]] for p in range(nSpins-nRows+1) ]
pvcons_pairs = [ [1+p, 1+p+2*nRows, pvcons[p]] for p in range(nSpins-2*nRows) ]

data = np.array(hcons_pairs+vcons_pairs+phcons_pairs+pvcons_pairs)
data = data[np.all(data != 0.0, axis=1)]  # get rid of zero rows
np.savetxt('inst_'+str(inst)+'.txt', data, fmt='%1d %1d %.12f ')
    
