# encoding: utf-8
# cython: profile=False
# filename: qmc.pyx
'''

File: qmc.py
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do path-integral quantum annealing.
             See: 10.1103/PhysRevB.66.094203

'''

cimport cython
import numpy as np
cimport numpy as np
cimport openmp
from cython.parallel import prange
from libc.math cimport exp as cexp
from libc.math cimport tanh as ctanh
from libc.math cimport log as clog
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX
# from libc.stdio cimport printf as cprintf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnneal(np.float_t[:] sched,
                    int mcsteps,
                    int slices, 
                    float temp, 
                    int nspins, 
                    np.float_t[:, :] confs, 
                    np.float_t[:, :, :] nbs,
                    rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    The Hamiltonian is:

    H = -\sum_k^P( \sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)). The second term on the RHS is a 
    1D Ising chain along the extra dimension. In other words, a spin in 
    this Trotter slice is coupled to that same spin in the nearest-neighbor
    slices.

    The quantum annealing is controlled by the strength of the transverse 
    field. This is given as an array of field values in @sched. @confs 
    stores the spin configurations for each replica which are updated 
    sequentially.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef float field = 0.0
    cdef float jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int sidx = 0
    cdef int tidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in xrange(sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(sched[ifield]/(slices*temp)))
        for step in xrange(mcsteps):
            # Loop over Trotter slices
            for islice in xrange(slices):
                # Loop over spins
                for sidx in sidx_shuff:
                    # loop through the neighbors
                    for si in xrange(maxnb):
                        # get the neighbor spin index
                        spinidx = int(nbs[sidx, si, 0])
                        # get the coupling value to that neighbor
                        jval = nbs[sidx, si, 1]
                        # self-connections are not quadratic
                        if spinidx == sidx:
                            ediff += -2.0*confs[sidx, islice]*jval
                        else:
                            ediff += -2.0*confs[sidx, islice]*(
                                jval*confs[spinidx, islice]
                            )
                    # periodic boundaries
                    if tidx == 0:
                        tleft = slices-1
                        tright = 1
                    elif tidx == slices-1:
                        tleft = slices-2
                        tright = 0
                    else:
                        tleft = islice-1
                        tright = islice+1
                    # now calculate between neighboring slices
                    ediff += -2.0*confs[sidx, islice]*(
                        jperp*confs[sidx, tleft])
                    ediff += -2.0*confs[sidx, islice]*(
                        jperp*confs[sidx, tright])
                    # Accept or reject
                    if ediff > 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                # reset energy diff
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnneal_dense(np.float_t[:] sched,
                          int mcsteps,
                          int slices, 
                          float temp, 
                          int nspins, 
                          np.float_t[:, :] confs, 
                          np.float_t[:, :] J,
                          rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    The Hamiltonian is:

    H = -\sum_k^P( \sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)). The second term on the RHS is a 
    1D Ising chain along the extra dimension. In other words, a spin in 
    this Trotter slice is coupled to that same spin in the nearest-neighbor
    slices.

    The quantum annealing is controlled by the strength of the transverse 
    field. This is given as an array of field values in @sched. @confs 
    stores the spin configurations for each replica which are updated 
    sequentially.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @J (np.ndarray, float): coupling matrix where off-diagonals
                                store coupling values and diagonal
                                stores local field biases
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int ifield = 0
    cdef float field = 0.0
    cdef float jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int sidx = 0
    cdef int tidx = 0
    cdef int si = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))
    # Loop over transverse field annealing schedule
    for ifield in xrange(sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(sched[ifield]/(slices*temp)))
        for step in xrange(mcsteps):
            # Loop over Trotter slices
            for islice in xrange(slices):
                # Loop over spins
                for sidx in sidx_shuff:
                    # loop through the neighbors
                    for si in xrange(nspins):
                        # self-connections are not quadratic
                        if si == sidx:
                            ediff += -2.0*confs[sidx, islice]*J[sidx,si]
                        else:
                            # incase we only have upper triangle
                            if sidx < si:
                                ediff += -2.0*confs[sidx, islice]*(
                                    J[sidx,si]*confs[si, islice]
                                )
                            else:
                                ediff += -2.0*confs[sidx, islice]*(
                                    J[si,sidx]*confs[si, islice]
                                )
                    # periodic boundaries
                    if tidx == 0:
                        tleft = slices-1
                        tright = 1
                    elif tidx == slices-1:
                        tleft = slices-2
                        tright = 0
                    else:
                        tleft = islice-1
                        tright = islice+1
                    # now calculate between neighboring slices
                    ediff += -2.0*confs[sidx, islice]*(
                        jperp*confs[sidx, tleft])
                    ediff += -2.0*confs[sidx, islice]*(
                        jperp*confs[sidx, tright])
                    # Accept or reject
                    if ediff > 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                # reset energy diff
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef QuantumAnneal_parallel(np.float_t[:] sched,
                             int mcsteps,
                             int slices, 
                             float temp, 
                             int nspins, 
                             np.float_t[:, :] confs, 
                             np.float_t[:, :, :] nbs,
                             int nthreads):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    The Hamiltonian is:

    H = -\sum_k^P( \sum_ij J_ij s^k_i s^k_j + J_perp \sum_i s^k_i s^k+1_i )

    where J_perp = -PT/2 log(tanh(G/PT)). The second term on the RHS is a 
    1D Ising chain along the extra dimension. In other words, a spin in 
    this Trotter slice is coupled to that same spin in the nearest-neighbor
    slices.

    The quantum annealing is controlled by the strength of the transverse 
    field. This is given as an array of field values in @sched. @confs 
    stores the spin configurations for each replica which are updated 
    sequentially.

    This version uses straightforward OpenMP threading to parallelize
    over inner spin-update loop.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @slices (int): number of replicas
        @temp (float): ambient temperature
        @confs (np.ndarray, float): contains the starting configurations
                                    for all Trotter replicas
        @nbs (np.ndarray, float): 3D array whose 1st dimension indexes
                                  each spin, 2nd dimension indexes
                                  neighbors to some spin, and 3rd
                                  dimension indexes the spin index
                                  of that neighbor (first element)
                                  or the coupling value to that
                                  neighbor (second element). See
                                  tools.GenerateNeighbors().
        @rng (np.RandomState): numpy random number generator object
        @nthreads (int): number of threads to execute in parallel

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int maxnb = nbs[0].shape[0]
    cdef int ifield = 0
    cdef float field = 0.0
    cdef float jperp = 0.0
    cdef int step = 0
    cdef int islice = 0
    cdef int sidx = 0
    cdef int tidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef int tleft = 0
    cdef int tright = 0
    # only reason we don't use memoryview is because we need arr.fill()
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(nspins)
    # Loop over transverse field annealing schedule
    for ifield in xrange(sched.size):
	# Calculate new coefficient for 1D Ising J
        jperp = -0.5*slices*temp*clog(ctanh(sched[ifield]/(slices*temp)))
        for step in xrange(mcsteps):
            # Loop over Trotter slices
            for islice in xrange(slices):
                # Loop over spins
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
                            ediffs[sidx] += -2.0*confs[sidx, islice]*jval
                        else:
                            ediffs[sidx] += -2.0*confs[sidx, islice]*(
                                jval*confs[spinidx, islice]
                            )
                    # periodic boundaries
                    if tidx == 0:
                        tleft = slices-1
                        tright = 1
                    elif tidx == slices-1:
                        tleft = slices-2
                        tright = 0
                    else:
                        tleft = islice-1
                        tright = islice+1
                    # now calculate between neighboring slices
                    ediffs[sidx] += -2.0*confs[sidx, islice]*(
                        jperp*confs[sidx, tleft])
                    ediffs[sidx] += -2.0*confs[sidx, islice]*(
                        jperp*confs[sidx, tright])
                    # Accept or reject
                    if ediffs[sidx] > 0.0:  # avoid overflow
                        confs[sidx, islice] *= -1
                    elif cexp(ediffs[sidx]/temp) > crand()/float(RAND_MAX):
                        confs[sidx, islice] *= -1
                # reset
                ediffs.fill(0.0)
