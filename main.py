'''

File: main.py
Author: Hadayat Seddiqi
Date: 10.07.14
Description: Do everything (will break this up later, if needed)

'''

import numpy as np
import scipy.sparse as sps

import pyximport; pyximport.install()
import sa
import qmc


#
# Initialize some parameters
#

# Do a classical annealing from a random start to the starting QMC
# temperature. Set to False for pure random start to quantum simulation.
preAnnealing = True
# MC steps per spin for pre-annealing stage
preAnnealingSteps = 1
# Pre-annealing initial temperature
preAnnealingTemperature = 3.0

# Number of Trotter slices
trotterSlices = 20
# Ambient temperature
annealingTemperature = 0.01e0
# Number of MC steps in actual annealing
annealingSteps = 100
# Transverse field starting strength
transFieldStart = 1.5
# Transverse field end
transFieldEnd = 1e-8

# Random number generator
rng = np.random.RandomState(1234)
# rng = np.random.RandomState()


#
# Generate a 2D square Ising model on a torus (with periodic boundaries)
#

# Number of rows in 2D square Ising model
nRows = 40
nSpins = nRows**2

# Horizontal nearest-neighbor couplings
hcons = rng.uniform(low=-2, high=2, size=nSpins)
hcons[::nRows] = 0.0

# Vertical nearest-neighbor couplings
vcons = rng.uniform(low=-2, high=2, size=nSpins)

# Horizontal periodic couplings
phcons = np.zeros(nSpins-2)
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


#
# Pre-annealing stage:
#
# Start with an initial random configuration at @initTemperature 
# and perform classical annealing down to @temperature to obtain 
# the initial configuration across all Trotter slices for QMC.
#

def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return [ 1 if k == 1 else -1 for k in vec ]


# Random initial configuration of spins
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nSpins) ], 
                      dtype=np.float)

print "Initial energy: ", sa.ClassicalIsingEnergy(spinVector, isingJ)

# Do the pre-annealing
if preAnnealing:
    sa.Anneal(preAnnealingTemperature, annealingTemperature,
              preAnnealingSteps, spinVector, isingJ, rng)

print "Final pre-annealing energy: ", sa.ClassicalIsingEnergy(spinVector, isingJ)


#
# Quantum Monte Carlo:
#
# Copy pre-annealed configuration as the initial configuration for all
# Trotter slices and carry out the true quantum annealing dynamics.
#

# Copy spin system over all the Trotter slices
# configurations = [ spinVector.copy() for k in xrange(trotterSlices) ]
# configurations = [ np.array(bits2spins(rng.random_integers(0, 1, nSpins))) 
#                    for k in xrange(trotterSlices) ]
# Rows are spin indices, columns represent Trotter slices
configurations = np.tile(spinVector, (trotterSlices, 1)).T

# Create 1D Ising matrix corresponding to extra dimension
perpJ = sps.dia_matrix(([[-trotterSlices*annealingTemperature/2.], 
                         [-trotterSlices*annealingTemperature/2.]], 
                        [1, trotterSlices-1]), 
                       shape=(trotterSlices, trotterSlices))

# Calculate number of steps to decrease transverse field
transFieldStep = ((transFieldStart-transFieldEnd)/annealingSteps)

# Execute quantum annealing part
qmc.QuantumAnneal(transFieldStart, transFieldStep, annealingSteps, 
                  trotterSlices, annealingTemperature, nSpins, perpJ, isingJ,
                  configurations, rng)

energies = [ sa.ClassicalIsingEnergy(c, isingJ) for c in configurations.T ]
print "Final quantum annealing energy: "
print "Lowest: ", np.min(energies)
print "Highest: ", np.max(energies)
print "Average: ", np.average(energies)
print "All: "
print energies
