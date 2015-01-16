# encoding: utf-8
# cython: profile=False
# filename: sa.pyx
'''

File: sa.py
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do the thermal pre-annealing in Cython.

'''

import numpy as np
cimport numpy as np
cimport cython

@cython.embedsignature(True)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def ClassicalMetropolisAccept(rng, 
                              np.ndarray[np.float_t, ndim=1] svec, 
                              int sidx, 
                              np.ndarray[np.float_t, ndim=2] nb_pairs, 
                              float T):
    """
    The Metropolis rule is given by accepting a proposed move s0 -> s1
    with an acceptance probability of:
    
    P(s0, s1) = min{1, exp[E(s0) - E(s1)]} .

    Input: @svec      spin vector
           @sidx      spin move to be accepted or rejected
           @nb_pairs  2D array of neighbor index-coupling value pairs
           @T         ambient temperature

    Returns: True  if move is accepted
             False if rejected
    """
    # define with cdefs to speed things up
    cdef float ediff = 0.0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    # loop through the neighbors
    for si in range(len(nb_pairs)):
        # get the neighbor spin index
        spinidx = nb_pairs[si][0]
        # get the coupling value to that neighbor
        jval = nb_pairs[si][1]
        # self-connections are not quadratic
        if spinidx == sidx:
            ediff += -2.0*svec[sidx]*jval
        else:
            ediff += -2.0*svec[sidx]*(jval*svec[int(spinidx)])
    # Accept or reject (if it's greater than the random sample, it
    # will always be accepted since it's bounded by [0, 1]).
    if ediff > 0.0:  # avoid overflow
        return True
    if np.exp(ediff/T) > rng.uniform(0,1):
        return True
    else:
        return False

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def Anneal(np.ndarray[np.float_t, ndim=1] annealingSchedule, 
           int mcSteps, 
           np.ndarray[np.float_t, ndim=1] spinVector, 
           np.ndarray[np.float_t, ndim=3] neighbors, 
           rng):
    """
    Execute thermal annealing according to @annealingSchedule, an
    array of temperatures, which takes @mcSteps number of Monte Carlo
    steps per timestep.

    Starting configuration is given by @spinVector, which we update 
    and calculate energies using the Ising graph @isingJ. @rng is the 
    random number generator.

    Returns: None (spins are flipped in-place)
    """
    # Number of spins
    nSpins = spinVector.size
    # Loop over temperatures
    for temp in annealingSchedule:
        # Do some number of Monte Carlo steps
        for step in xrange(mcSteps):
            # Loop over spins
            for idx in rng.permutation(range(nSpins)):
                # Attempt to flip this spin
                if ClassicalMetropolisAccept(rng, spinVector, idx, 
                                             neighbors[idx], temp):
                    spinVector[idx] *= -1
