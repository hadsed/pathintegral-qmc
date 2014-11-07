'''

File: main.py
Author: Hadayat Seddiqi
Date: 10.07.14
Description: Do everything (will break this up later, if needed)

'''

import numpy as np
import scipy.constants as spconst


#
# Initialize some parameters
#

# Boltzmann constant
kboltz = spconst.k

# MC steps per spin for pre-annealing stage to set initial config
preAnnealingSteps = 50
# Pre-annealing initial temperature
preAnnealingTemperature = 3.0

# Number of MC steps in actual annealing
annealingSteps = 100
# Number of Trotter slices
trotterSlices = 20
# Ambient temperature
annealingTemperature = 0.01

# Transverse field starting strength
transFieldStart = 1.5
# Transverse field end
transFieldEnd = 1e-8
# Step size for linear annealing schedule
transFieldStep = 0.1

# Number of spins in 2D Ising model
nSpins = 20
# Random number generator
rng = np.random.RandomState(1234)
# The quantum Ising coupling matrix
isingJ = rng.uniform(low=-2, high=2, size=(nSpins, nSpins))


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

def MetropolisAccept(svec, fidx, J, B):
    """
    The Metropolis rule is given by accepting a proposed move s0 -> s1
    with an acceptance probability of:
    
    P(s0, s1) = min{1, exp[E(s0) - E(s1)]} .

    Input: @svec is the spin vector
           @fidx is the spin move to be accepted or rejected
           @J is the Ising coupling matrix
           @B is beta, i.e., (kboltz*temperature)^-1

    Returns: True if move is accepted
             False if rejected
    """
    s0 = np.matrix(svec, dtype=np.float64).T
    s1 = s0.copy()
    s1[fidx] *= -1
    energyDiff = np.exp(s0.T*J*s0 - s1.T*J*s1)
    # Accept or reject (if it's greater than the random sample, it
     # will always be accepted since it's bounded by [0, 1]).
    if energyDiff > np.random.uniform(0,1):
        return True
    else:
        return False

# Random initial configuration of spins
spinVector = bits2spins(np.random.random_integers(0, 1, nSpins))
print spinVector
print np.matrix(spinVector)*isingJ*np.matrix(spinVector).T
# Track accepted moves
accepted = 0
# How much to reduce the temperature at each step
instTempStep = (preAnnealingTemperature - annealingTemperature) \
    /float(preAnnealingSteps)

# Loop over temperatures
for temp in (preAnnealingTemperature - k*instTempStep 
             for k in xrange(preAnnealingSteps+1)):
    # Loop over pre-annealing steps
    for step in xrange(preAnnealingSteps):
        # Loop over spins
        for idx, spin in enumerate(spinVector):
            # Attempt to flip this spin
            if MetropolisAccept(spinVector, idx, isingJ, 
                                1./(kboltz*temp)):
                spinVector[idx] *= -1
                accepted += 1  # track acceptances
print "Final temperature: ", temp
print "Final state: ", spinVector
print "Number of accepted moves: ", accepted
print "Final energy: ", np.matrix(spinVector)*isingJ*np.matrix(spinVector).T
