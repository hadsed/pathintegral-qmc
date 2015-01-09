'''

File: sa.py
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do the thermal pre-annealing in Cython.

'''

import numpy as np
cimport numpy as np


def ClassicalIsingEnergy(spins, J):
    """
    Calculate energy for Ising graph @J in configuration @spins.
    Generally not needed for the annealing process but useful to
    have around at the end of simulations.
    """
    J = np.asarray(J.todense())
    d = np.diag(np.diag(J))
    np.fill_diagonal(J, 0.0)
    return -np.dot(spins, np.dot(J, spins)) - np.sum(np.dot(d,spins))

def ClassicalMetropolisAccept(rng, np.ndarray[np.float_t, ndim=1] svec, 
                              int fidx, nb_pairs, T):
    """
    The Metropolis rule is given by accepting a proposed move s0 -> s1
    with an acceptance probability of:
    
    P(s0, s1) = min{1, exp[E(s0) - E(s1)]} .

    Input: @svec   spin vector
           @fidx   spin move to be accepted or rejected
           @J      Ising coupling matrix
           @T      ambient temperature

    Returns: True  if move is accepted
             False if rejected
    """
    # calculate energy difference
    ediff = 0.0
    for spinidx, jval in nb_pairs:
        # self-connections are not quadratic
        if spinidx == fidx:
            ediff += -2.0*svec[fidx]*jval
        else:
            ediff += -2.0*svec[fidx]*(jval*svec[spinidx])
    # Accept or reject (if it's greater than the random sample, it
    # will always be accepted since it's bounded by [0, 1]).
    if ediff > 0.0:
        return True
    if np.exp(ediff/T) > rng.uniform(0,1):
        return True
    else:
        return False

def Anneal(float preAnnealingTemperature, float annealingTemperature, 
           int preAnnealingSteps, 
           np.ndarray[np.float_t, ndim=1] spinVector, neighbors, rng):
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
                                             neighbors[idx], temp):
                    spinVector[idx] *= -1
