# encoding: utf-8
# cython: profile=False
# filename: sa.pyx
'''

File: sa.pyx
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do thermal annealing on a (sparse) Ising system.

'''

import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange
from libc.math cimport exp as cexp
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX
# from libc.stdio cimport printf as cprintf


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
cpdef Anneal(np.float_t[:] sched, 
             int mcsteps, 
             np.float_t[:] svec, 
             np.float_t[:, :, :] nbs, 
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
    # Define some variables
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))

    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            for sidx in sidx_shuff:
                # loop through the given spin's neighbors
                for si in xrange(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[sidx,si,0])
                    # get the coupling value to that neighbor
                    jval = nbs[sidx,si,1]
                    # self-connections are not quadratic
                    if spinidx == sidx:
                        ediff += -2.0*svec[sidx]*jval
                    # calculate the energy diff of flipping this spin
                    else:
                        ediff += -2.0*svec[sidx]*(jval*svec[spinidx])
                # Metropolis accept or reject
                if ediff > 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef Anneal_parallel(np.float_t[:] sched, 
                      int mcsteps, 
                      np.float_t[:] svec, 
                      np.float_t[:, :, :] nbs, 
                      int nthreads):
    """
    Execute thermal annealing according to @annealingSchedule, an
    array of temperatures, which takes @mcSteps number of Monte Carlo
    steps per timestep.

    Starting configuration is given by @spinVector, which we update 
    and calculate energies using the Ising graph @isingJ.

    This version attempts to do thread parallelization with Cython's
    built-in OpenMP directive "prange". The extra argument @nthreads
    specifies how many workers to split the spin updates amongst.

    Note that while the sequential version randomizes the order of
    spin updates, this version does not.

    Returns: None (spins are flipped in-place)
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(nspins)

    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            # print nthreads, openmp.omp_get_num_threads()
            for sidx in prange(nspins, nogil=True, 
                               schedule='guided', 
                               num_threads=nthreads):
                # loop through the neighbors
                for si in xrange(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[sidx, si, 0])
                    # get the coupling value to that neighbor
                    jval = nbs[sidx, si, 1]
                    # self-connections are not quadratic
                    if spinidx == sidx:
                        ediffs[sidx] += -2.0*svec[sidx]*jval
                    else:
                        ediffs[sidx] += -2.0*svec[sidx]*(jval*svec[spinidx])
                # Accept or reject
                if ediffs[sidx] > 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediffs[sidx]/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
            # reset
            ediffs.fill(0.0)
