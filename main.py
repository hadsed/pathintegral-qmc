'''

File: main.py
Author: Hadayat Seddiqi
Date: 10.07.14
Description: Do everything (will break this up later, if needed)

'''

import numpy as np
import scipy.sparse as sps

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
nRows = 8
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

def ClassicalIsingEnergy(spins, J):
    """ Calculate energy for Ising graph @J in configuration @spins. """
    return -np.dot(spins, J.dot(spins))

def ClassicalMetropolisAccept(rng, svec, fidx, J, T):
    """
    The Metropolis rule is given by accepting a proposed move s0 -> s1
    with an acceptance probability of:
    
    P(s0, s1) = min{1, exp[E(s0) - E(s1)]} .

    Input: @svec is the spin vector
           @fidx is the spin move to be accepted or rejected
           @J is the Ising coupling matrix
           @T is the ambient temperature

    Returns: True if move is accepted
             False if rejected
    """
    e0 = ClassicalIsingEnergy(svec, J)
    svec[fidx] *= -1
    e1 = ClassicalIsingEnergy(svec, J)
    svec[fidx] *= -1  # we're dealing with the original array, so flip back
    # Accept or reject (if it's greater than the random sample, it
    # will always be accepted since it's bounded by [0, 1]).
    if (e0 - e1) > 0.0:
        return True
    if np.exp((e0 - e1)/T) > rng.uniform(0,1):
        return True
    else:
        return False

# Random initial configuration of spins
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nSpins) ])
print "Initial energy: ", ClassicalIsingEnergy(spinVector, isingJ)

if preAnnealing:
    # How much to reduce the temperature at each step
    instTempStep = (preAnnealingTemperature - annealingTemperature) \
        /float(preAnnealingSteps)
    # Loop over temperatures
    for temp in (preAnnealingTemperature - k*instTempStep 
                 for k in xrange(preAnnealingSteps+1)):
        # Do some number of Monte Carlo steps
        for step in xrange(preAnnealingSteps):
            # Loop over spins
            for idx in rng.permutation(range(spinVector.size)):
                # Attempt to flip this spin
                if ClassicalMetropolisAccept(rng, spinVector, idx, 
                                             isingJ, temp):
                    spinVector[idx] *= -1

# print "Final temperature: ", temp
# print "Final state: ", spinVector
print "Final pre-annealing energy: ", ClassicalIsingEnergy(spinVector, isingJ)


#
# Quantum Monte Carlo:
#
# Copy pre-annealed configuration as the initial configuration for all
# Trotter slices and carry out the true quantum annealing dynamics.
#

def QuantumIsingEnergy(spins, tspins, J, Jperp):
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
    firstTerm = np.dot(spins, J.dot(spins))
    secondTerm = np.dot(tspins, Jperp.dot(tspins))
    return -tspins.size*(firstTerm+secondTerm)

def QuantumMetropolisAccept(rng, svec, fidx, tvec, J, Jperp, T):
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
    e0 = QuantumIsingEnergy(svec, tvec, J, Jperp)
    svec[fidx] *= -1
    e1 = QuantumIsingEnergy(svec, tvec, J, Jperp)
    svec[fidx] *= -1  # we're dealing with the original array, so flip back
    if (e0 - e1) > 0.0:  # avoid overflow
        return True
    if np.exp((e0 - e1)/T) > rng.uniform(0,1):
        return True
    else:
        return False

# Copy spin system over all the Trotter slices
configurations = [ spinVector.copy() for k in xrange(trotterSlices) ]
# configurations = [ np.array(bits2spins(rng.random_integers(0, 1, nSpins))) 
#                    for k in xrange(trotterSlices) ]

# Create 1D Ising matrix corresponding to extra dimension
perpJ = sps.dia_matrix(([[-trotterSlices*annealingTemperature/2.], 
                         [-trotterSlices*annealingTemperature/2.]], 
                        [1, trotterSlices-1]), 
                       shape=(trotterSlices, trotterSlices))

# Calculate number of steps to decrease transverse field
transFieldStep = ((transFieldStart-transFieldEnd)/annealingSteps)

# Loop over transverse field annealing schedule
for field in (transFieldStart - k*transFieldStep 
              for k in xrange(annealingSteps+1)):
    # Calculate new coefficient for 1D Ising J
    perpJCoeff = np.log(np.tanh(field/(trotterSlices*annealingTemperature)))
    calculatedPerpJ = perpJCoeff*perpJ
    # Loop over Trotter slices
    for islice in rng.permutation(range(trotterSlices)):
        # print "Trotter slice: ", islice
        # Loop over spins
        for ispin in rng.permutation(range(nSpins)):
            # Grab nearest-neighbor spin vector across Trotter slices
            trotterSpins = np.array([ vec[ispin] for vec in configurations ])
            # Attempt to flip this spin
            if QuantumMetropolisAccept(rng, configurations[islice], ispin, 
                                       trotterSpins, isingJ, 
                                       calculatedPerpJ, annealingTemperature):
                configurations[islice][ispin] *= -1

print "Final quantum annealing energy: "
energies = [ ClassicalIsingEnergy(c, isingJ) for c in configurations ]
print "Lowest: ", np.min(energies)
print "Highest: ", np.max(energies)
print "Average: ", np.average(energies)
print "All: "
print energies
