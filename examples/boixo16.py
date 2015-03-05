'''

File: boixo16.py
Author: Hadayat Seddiqi
Date: 03.05.15
Description: Run the 16-qubit double bipartite graph 
             problem from Boixo et al (2015).

'''

import numpy as np
import scipy.sparse as sps

import piqmc.sa as sa
import piqmc.qmc as qmc
import piqmc.tools as tools

# Define some parameters
nspins = 16
preannealingsteps = 100
preannealingmcsteps = 1
preannealingtemp = 10.0
annealingtemp = 0.01
annealingsteps = 1000
annealingmcsteps = 1
trotterslices = 20
fieldstart = 10.0
fieldend = 1e-8
samples = 100
# Random number generator
seed = 1234
rng = np.random.RandomState(seed)
# Test file name
inputfname = 'ising_instances/boixo16.txt'

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
hleft = 0.44
for k in xrange(8):
    isingJ[k,k] = hleft
# # Print out energies we're supposed to see from QMC sims
# print("All possible states and their energies:")
# results = []
# def bitstr2spins(vec):
#     """ Take a bitstring and return a spinvector. """
#     a = [ int(k) for k in vec ]
#     return tools.bits2spins(a)
# for b in [ bin(x)[2:].rjust(nspins, '0') for x in range(2**nspins) ]:
#     bvec = np.array([ int(k) for k in b ])
#     svec = bitstr2spins(b)
#     bstr = reduce(lambda x,y: x+y, [ str(k) for k in bvec ])
#     results.append([sa.ClassicalIsingEnergy(svec, isingJ), bstr])
# for res in sorted(results)[:100]:
#     print res

# Initialize random state
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                      dtype=np.float)
confs = np.tile(spinVector, (trotterslices, 1)).T
# Generate list of nearest-neighbors for each spin
neighbors = tools.GenerateNeighbors(nspins, isingJ, 6)
# Generate annealing schedules
tannealingsched = np.linspace(preannealingtemp,
                              annealingtemp,
                              annealingsteps)
annealingsched = np.linspace(fieldstart,
                             fieldend,
                             annealingsteps)
# keep a count of the populations
gstate_sa = 0
exstate_sa = 0
sa_errors = 0
gstate_qmc = 0
exstate_qmc = 0
qmc_errors = 0
qmc_errors_diff = 0

print("")
print("# samples", samples)

# Try using SA (random start)
for sa_itr in xrange(samples):
    spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                          dtype=np.float)
    starten, startstate = (sa.ClassicalIsingEnergy(spinVector, isingJ), 
                           getbitstr(spinVector))
    sa.Anneal(tannealingsched, preannealingmcsteps, 
              spinVector, neighbors, rng)
    bitstr = getbitstr(spinVector)
    # print("Start", starten, startstate[:8], startstate[8:])
    # print(#sa.ClassicalIsingEnergy(spinVector, isingJ), 
    #       bitstr[:8], bitstr[8:])
    if bitstr == '0000000011111111':
        exstate_sa += 1
    elif(bitstr == '1111111111111111' or
         bitstr == '0000000000000000'):
        gstate_sa += 1
    else:
        sa_errors += 1
        
# gstate_sa /= float(samples)
# exstate_sa /= float(samples)
print("")
print("SA ground:", gstate_sa)
print("SA excited:", exstate_sa)
print("SA errors:", sa_errors)

# Now do PIQA
for s in xrange(samples):
    confs = np.tile(np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                             dtype=np.float), (trotterslices, 1)).T
    qmc.QuantumAnneal(annealingsched, annealingmcsteps, 
                      trotterslices, annealingtemp, nspins, 
                      confs, neighbors, rng)
    if not np.all(np.sum(confs, axis=1)/trotterslices
                  == confs[:,0]):
        print("error")
        qmc_errors_diff += 1
    else:
        bitstr = reduce(
            lambda x,y: x+y, 
            [ str(int(k)) for k in tools.spins2bits(confs[:,0]) ]
        )
        # print(bitstr[:8],bitstr[8:])
        if bitstr == '0000000011111111':
            exstate_qmc += 1
        elif(bitstr == '1111111111111111' or
             bitstr == '0000000000000000'):
            gstate_qmc += 1
        else:
            qmc_errors += 1

# gstate_qmc /= float(samples)
# exstate_qmc /= float(samples)
print("")
print("QMC ground:", gstate_qmc)
print("QMC excited:", exstate_qmc)
print("QMC higher excitations:", qmc_errors)
print("QMC differing slices:", qmc_errors_diff)
