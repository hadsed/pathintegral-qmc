'''

File: test_boixo.py
Author: Hadayat Seddiqi
Date: 01.06.15
Description: Test the accuracy of PIQA code by running the
             8-qubit diamond graph problem from Boixo et al.

'''

import pickle
import numpy as np
import scipy.sparse as sps
import piqa

# Define some parameters
nspins = 8
preannealingsteps = 2
preannealingmcsteps = 1
preannealingtemp = 1.0
annealingtemp = 0.01
annealingsteps = 10
annealingmcsteps = 1
trotterslices = 5
fieldstart = 0.5
fieldend = 1e-8
# Random number generator
seed = None
rng = np.random.RandomState(seed)
# Test file name
inputfname = 'ising_instances/boixo.txt'

def getbitstr(vec):
    """ Return bitstring from spin vector array. """
    return reduce(lambda x,y: x+y, 
                  [ str(int(k)) for k in piqa.spins2bits(vec) ])

# Read from textfile directly to be sure
loaded = np.loadtxt(inputfname)
# Construct Ising matrix
isingJ = sps.dok_matrix((nspins,nspins))
for i,j,val in loaded:
    isingJ[i-1,j-1] = val

# Print out energies we're supposed to see from QMC sims
print("All possible states and their energies:")
results = []
def bitstr2spins(vec):
    """ Take a bitstring and return a spinvector. """
    a = [ int(k) for k in vec ]
    return piqa.bits2spins(a)
for b in [ bin(x)[2:].rjust(nspins, '0') for x in range(2**nspins) ]:
    bvec = np.array([ int(k) for k in b ])
    svec = bitstr2spins(b)
    bstr = reduce(lambda x,y: x+y, [ str(k) for k in bvec ])
    results.append([piqa.sa.ClassicalIsingEnergy(svec, isingJ), bstr])
for res in sorted(results):
    print res

# Initialize random state
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                      dtype=np.float)
spinVector_original = spinVector.copy()
configurations = np.tile(spinVector, (trotterslices, 1)).T
# Generate list of nearest-neighbors for each spin
neighbors = piqa.GenerateNeighbors(nspins, isingJ, 4)
# Generate annealing schedules
tannealingsched = np.linspace(preannealingtemp,
                              annealingtemp,
                              annealingsteps)
annealingsched = np.linspace(fieldstart,
                             fieldend,
                             annealingsteps)
# Try using SA (deterministic start)
print ("SA results using same starting state:")
print ("Starting state: ",
       piqa.sa.ClassicalIsingEnergy(spinVector, isingJ),
       getbitstr(spinVector))
for sa_itr in range(trotterslices):
    spinVector = spinVector_original.copy()
    piqa.sa.Anneal(tannealingsched, preannealingmcsteps, 
                   spinVector, neighbors, rng)
    print(piqa.sa.ClassicalIsingEnergy(spinVector, isingJ), 
          getbitstr(spinVector))
# Try using SA (random start)
print ("SA results using random state (start and end):")
for sa_itr in range(trotterslices):
    spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                          dtype=np.float)
    starten, startstate = (piqa.sa.ClassicalIsingEnergy(spinVector, isingJ), 
                           getbitstr(spinVector))
    piqa.sa.Anneal(tannealingsched, preannealingmcsteps, 
                   spinVector, neighbors, rng)
    print(starten, startstate, 
          piqa.sa.ClassicalIsingEnergy(spinVector, isingJ), 
          getbitstr(spinVector))

# Now do PIQA
print ("QA results:")
print("PIQA starting state: ", getbitstr(spinVector))
piqa.qmc.QuantumAnneal(annealingsched, annealingmcsteps, 
                       trotterslices, annealingtemp, nspins, 
                       configurations, neighbors, rng)
minEnergy, minConfiguration = np.inf, []
print ("Final states of PIQA replicas: ")
for col in configurations.T:
    candidateEnergy = piqa.sa.ClassicalIsingEnergy(col, isingJ)
    print(candidateEnergy, 
          reduce(lambda x,y: x+y, 
                 [ str(int(k)) for k in piqa.spins2bits(col) ]))
    if candidateEnergy < minEnergy:
        minEnergy = candidateEnergy
        minConfiguration = col
