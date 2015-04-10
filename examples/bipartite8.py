'''

File: bipartite8.py
Author: Hadayat Seddiqi
Date: 04.07.15
Description: Test the bipartite solver for simulated annealing.

'''

import numpy as np
import scipy.sparse as sps

import piqmc.sa as sa
import piqmc.tools as tools

# Define some parameters
nspins = 8
preannealingtemp = 1.1
annealingtemp = 0.1
annealingsteps = 10
annealingmcsteps = 1
samples = 20
# Random number generator
seed = None
rng = np.random.RandomState(seed)
# Test file name
inputfname = 'ising_instances/bipartite8.txt'

def getbitstr(vec):
    """ Return bitstring from spin vector array. """
    return reduce(lambda x,y: x+y, 
                  [ str(int(k)) for k in tools.spins2bits(vec) ])

# Read from textfile directly to be sure
loaded = np.loadtxt(inputfname)
# Construct Ising matrix
isingJ = sps.dok_matrix((nspins,nspins))
for i,j,val in loaded:
    isingJ[i-1,j-1] = val
# bias the left cluster
# hleft = 0.44
# for k in xrange(8):
#     isingJ[k,k] = hleft
# Print out energies we're supposed to see
print("All possible states and their energies:")
results = []
def bitstr2spins(vec):
    """ Take a bitstring and return a spinvector. """
    a = [ int(k) for k in vec ]
    return tools.bits2spins(a)
for b in [ bin(x)[2:].rjust(nspins, '0') for x in range(2**nspins) ]:
    bvec = np.array([ int(k) for k in b ])
    svec = bitstr2spins(b)
    bstr = reduce(lambda x,y: x+y, [ str(k) for k in bvec ])
    results.append([sa.ClassicalIsingEnergy(svec, isingJ), bstr])
for res in sorted(results):#[:100]:
    print res
print("\n")
# Generate list of nearest-neighbors for each spin
neighbors = tools.GenerateNeighbors(nspins, isingJ, 5)
# Generate annealing schedules
tannealingsched = np.linspace(preannealingtemp,
                              annealingtemp,
                              annealingsteps)
# Generate random states to compare
svecs = np.asarray([ [ 2*rng.randint(2)-1 for k in range(nspins) ] 
                     for j in xrange(samples) ], 
                   dtype=np.float)
problems = ['11111000', '11000001', '01100101', 
            '01101000', '00010011', '11110100']
problems = np.array([ bitstr2spins(k) for k in problems ])
svecs = np.vstack((problems, svecs))

# Try using SA (random start)
for sa_itr in xrange(samples):
    # spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
    #                       dtype=np.float)
    spinVector = svecs[sa_itr].copy()
    starten, startstate = (sa.ClassicalIsingEnergy(spinVector, isingJ), 
                           getbitstr(spinVector))
    sa.Anneal(tannealingsched, annealingmcsteps, 
              spinVector, neighbors, rng)
    bitstr = getbitstr(spinVector)
    print("Start", starten, startstate[:8], 
          "End", sa.ClassicalIsingEnergy(spinVector, isingJ), 
          bitstr[:8])
print("")
# Now test the bipartite annealer
vnum, hnum = 4, 4
# extract diagonal
ds = np.diag(isingJ.todense())
vbias = ds[:vnum].copy()
hbias = ds[vnum:].copy()
# extract bipartite block
W = isingJ[:vnum,hnum:].todense()
# sample
for sa_itr in xrange(samples):
    spinVector = svecs[sa_itr].copy()
    vvec = spinVector[:vnum].copy()
    hvec = spinVector[vnum:].copy()
    starten, startstate = (sa.ClassicalIsingEnergy(spinVector, isingJ), 
                           getbitstr(spinVector))
    sa.Anneal_bipartite(tannealingsched,
                        annealingmcsteps,
                        vvec,
                        hvec,
                        W,
                        vbias,
                        hbias,
                        rng)
    spinVector = np.concatenate((vvec,hvec))
    bitstr = getbitstr(spinVector)
    print("Start", starten, startstate[:8], 
          "End", sa.ClassicalIsingEnergy(spinVector, isingJ), 
          bitstr[:8])
