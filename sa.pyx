'''

File: sa.py
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do the thermal pre-annealing in Cython.

'''

import numpy as np
cimport numpy as np


def ClassicalIsingEnergy(np.ndarray[np.float_t, ndim=1] spins, J):
    """ Calculate energy for Ising graph @J in configuration @spins. """
    return -np.dot(spins, J.dot(spins))

def ClassicalMetropolisAccept(rng, np.ndarray[np.float_t, ndim=1] svec, 
                              int fidx, J, T):
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

def Anneal(float preAnnealingTemperature, float annealingTemperature, 
           int preAnnealingSteps, 
           np.ndarray[np.float_t, ndim=1] spinVector, isingJ, rng):
    """
    Execute thermal annealing from @preAnnealingTemperature down to 
    @annealingTemperature with @preAnnealingSteps number of steps. 
    Starting configuration is given by @spinVector, which we update 
    using MC steps and calculate energies using the Ising graph @isingJ. 
    @rng is the random number generator.

    Returns: None (spins are flipped in-place)
    """
    # How much to reduce the temperature at each step
    instTempStep = (preAnnealingTemperature - annealingTemperature) \
        /preAnnealingSteps
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

def ClassicalMetropolisAccept_opt(rng, np.ndarray[np.float_t, ndim=1] svec, 
                                  int fidx, nb_pairs, T):
    """
    Optimized version.
    """
    # calculate energy difference
    ediff = 0.0
    for spinidx, jval in nb_pairs:
        ediff += -2.0*svec[fidx]*(jval*svec[spinidx])
    # Accept or reject (if it's greater than the random sample, it
    # will always be accepted since it's bounded by [0, 1]).
    if ediff > 0.0:
        return True
    if np.exp(ediff/T) > rng.uniform(0,1):
        return True
    else:
        return False

def Anneal_opt(float preAnnealingTemperature, float annealingTemperature, 
               int preAnnealingSteps, 
               np.ndarray[np.float_t, ndim=1] spinVector, isingJ, rng):
    """
    Optimized version.
    """
    # Precompute neighbors for each spin
    nspins = len(spinVector)
    nrows = int(np.sqrt(nspins))
    isingJ = isingJ.todok()  # dictionary of keys type makes this easy
    neighbors = []
    # Iterate over all spins
    for ispin, spin in enumerate(spinVector):
        nb_pairs = []
        # Find the pairs including this spin
        for pair in isingJ.iterkeys():
            if pair[0] == ispin:
                nb_pairs.append([ pair[1], isingJ[pair] ])
            elif pair[1] == ispin:
                nb_pairs.append([ pair[0], isingJ[pair] ])
        # Record it in the master list
        neighbors.append(nb_pairs)

    # How much to reduce the temperature at each step
    instTempStep = (preAnnealingTemperature - annealingTemperature) \
        /preAnnealingSteps
    # Loop over temperatures
    for temp in (preAnnealingTemperature - k*instTempStep 
                 for k in xrange(preAnnealingSteps+1)):
        # Do some number of Monte Carlo steps
        for step in xrange(preAnnealingSteps):
            # Loop over spins
            for idx in rng.permutation(range(spinVector.size)):
                # Attempt to flip this spin
                if ClassicalMetropolisAccept_opt(rng, spinVector, idx, 
                                                 neighbors[idx], temp):
                    spinVector[idx] *= -1
