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
isingJ = np.triu(rng.uniform(low=-2, high=2, size=(nSpins, nSpins)))


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

def ClassicalIsingEnergy(spins, J):
    """ Calculate energy for Ising graph @J in configuration @spins. """
    return np.dot(spins, np.dot(J, spins))

def ClassicalMetropolisAccept(svec, fidx, J, B):
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
    e0 = ClassicalIsingEnergy(svec, J)
    svec[fidx] *= -1
    e1 = ClassicalIsingEnergy(svec, J)
    svec[fidx] *= -1  # we're dealing with the original array, so flip back
    energyDiff = np.exp(e0 - e1)
    # Accept or reject (if it's greater than the random sample, it
    # will always be accepted since it's bounded by [0, 1]).
    if energyDiff > np.random.uniform(0,1):
        return True
    else:
        return False

# Random initial configuration of spins
spinVector = bits2spins(np.random.random_integers(0, 1, nSpins))
# print spinVector
print "Initial energy: ", ClassicalIsingEnergy(spinVector, isingJ)
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
            if ClassicalMetropolisAccept(spinVector, idx, isingJ, 
                                1./(kboltz*temp)):
                spinVector[idx] *= -1

# print "Final temperature: ", temp
# print "Final state: ", spinVector
print "Final energy: ", ClassicalIsingEnergy(spinVector, isingJ)


#
# Quantum Monte Carlo:
#
# Copy pre-annealed configuration as the initial configuration for all
# Trotter slices and carry out the true quantum annealing dynamics.
#

def QuantumIsingEnergy(spins, tspins, J, T, P, G):
    """
    Calculate the energy of the following Ising Hamiltonian with an 
    extra dimension along the Trotter slices:

    H = -\sum_k^P( \sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)). The second term on the RHS is a 
    1D Ising chain along the extra dimension. In other words, a spin in this
    Trotter slice is coupled to that same spin in the nearest-neighbor slices.

    Inputs: @spins is the spin vector
            @J is the original Ising coupling matrix
            @T is the bath temperature
            @P is the number of Trotter slices
            @G is the transverse field strength

    Returns: the energy as a float

    """
    firstTerm = np.dot(spins, np.dot(J, spins))
    secondTerm = np.zeros(J.shape)
    np.fill_diagonal(secondTerm, -P*T/2.)


def QuantumMetropolisAccept(svec, fidx, tvec, J, T, P, G):
    """
    Essentially the same as ClassicalMetropolisAccept(), except that
    we use a different calculation for the energies.

    Inputs: @svec is the spin vector
            @fidx is the spin we wish to flip
            @J is the original Ising coupling matrix
            @T is the bath temperature
            @P is the number of Trotter slices
            @G is the transverse field strength

    Returns: True if move is accepted
             False if rejected

    """
    e0 = QuantumIsingEnergy(svec, tvec, J, T, P, G)
    svec[fidx] *= -1
    e1 = QuantumIsingEnergy(svec, tvec, J, T, P, G)
    svec[fidx] *= -1  # we're dealing with the original array, so flip back
    energyDiff = np.exp(e0 - e1)
    if energyDiff > np.random.uniform(0,1):
        return True
    else:
        return False


# Copy spin system over all the Trotter slices
configurations = [ spinVector.copy() for k in xrange(trotterSlices) ]
# Loop over transverse field annealing schedule
for field in xrange(transFieldStart, transFieldEnd, transFieldStep):
    # Loop over Trotter slices
    for itslice, tslice in enumerate(configurations):
        # Loop over spins
        for ispin, spin in enumerate(xrange(nSpins)):
            # Grab nearest-neighbor spin vector across Trotter slices
            trotterSpins = [ vec[ispin] for vec in configurations ]
            # Attempt to flip this spin
            if QuantumMetropolisAccept(spinVector, idx, trotterSpins, isingJ)):
                spinVector[idx] *= -1
