# encoding: utf-8
# cython: profile=False
# filename: sa.pyx
'''

File: sa.pyx
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Thermal annealing routines for Ising models.

'''

import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange, parallel
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

    Args:
        @spins (np.array, float): configuration of spins (values +/-1)
        @J (np.ndarray, float): coupling matrix where off-diagonals
                                store coupling values and diagonal
                                stores local field biases

    Returns:
        float: the energy of configuration @spins in an Ising
               system specified by @J
    """
    J = np.asarray(J.todense())
    d = np.diag(np.diag(J))
    np.fill_diagonal(J, 0.0)
    return -np.dot(spins, np.dot(J, spins)) - np.sum(np.dot(d,spins))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal(np.float_t[:] sched, 
             int mcsteps, 
             np.float_t[:] svec, 
             np.float_t[:, :, :] nbs, 
             rng):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is 
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
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
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int schedsize = sched.size
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
    for itemp in xrange(schedsize):
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
                if ediff >= 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal_dense(np.float_t[:] sched, 
                   int mcsteps, 
                   np.float_t[:] svec, 
                   np.float_t[:, :] J, 
                   rng):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is 
    given by @svec, which we update in-place and calculate energies
    using the coupling matrix @J.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
        @J (np.ndarray, float): coupling matrix where off-diagonals
                                store coupling values and diagonal
                                stores local field biases
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
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
                for si in xrange(nspins):
                    # self-connections are not quadratic
                    if si == sidx:
                        ediff += -2.0*svec[sidx]*J[sidx,si]
                    # calculate the energy diff of flipping this spin
                    else:
                        # incase we only have upper triangle
                        if sidx < si:
                            ediff += -2.0*svec[sidx]*(J[sidx,si]*svec[si])
                        elif sidx > si:
                            ediff += -2.0*svec[sidx]*(J[si,sidx]*svec[si])
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
@cython.cdivision(True)
cpdef Anneal_parallel(np.float_t[:] sched, 
                      int mcsteps, 
                      np.float_t[:] svec, 
                      np.float_t[:, :, :] nbs, 
                      int nthreads):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is 
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    This version uses straightforward OpenMP threading to parallelize
    over inner spin-update loop.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
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
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(nspins)

    with nogil, parallel(num_threads=nthreads):
        # Loop over temperatures
        for itemp in xrange(schedsize):
            # Get temperature
            temp = sched[itemp]
            # Do some number of Monte Carlo steps
            for step in xrange(mcsteps):
                # Loop over spins
                # print nthreads, openmp.omp_get_num_threads()
                for sidx in prange(nspins, schedule='static'):
                    ediffs[sidx] = 0.0  # reset
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cdef inline bint getbit(np.uint64_t s, int k):
    """
    Get the @k-th bit of @s.
    """
    return (s >> k) & 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
def Anneal_multispin(np.float_t[:] sched, 
                     int mcsteps, 
                     np.float_t[:, :] svec_mat, 
                     np.float_t[:, :, :] nbs, 
                     rng):
    """
    Execute 64 simultaneous thermal annealing according to @sched 
    with @mcsteps sweeps for each annealing step. Starting configurations
    are given by @svec_mat, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    This version takes in a set of 64 configurations and packs them into
    integer arrays (since they are binary states) and uses bitwise
    operators to propagate the states simultaneously.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec_mat (np.ndarray, float): contains 64 starting configurations
                                       where each row denotes a starting
                                       configuration
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
        None: spins are flipped in-place within @svec_mat
    """
    # Define some variables
    cdef int nspins = svec_mat.shape[1]
    cdef int maxnb = nbs[0].shape[0]
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef int k = 0
    cdef np.uint64_t flipmask = 0
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(64)
    cdef np.ndarray[np.float_t, ndim=1] rands = rng.rand(64)
    cdef np.ndarray[np.uint64_t, ndim=1] svec = \
        np.zeros(nspins, dtype=np.uint64)
    cdef np.ndarray[np.int8_t, ndim=1] sign = \
        np.zeros(64, dtype=np.int8)
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))
    # encode @svec_mat into the bits of svec elements
    for si in xrange(svec_mat.shape[1]):
        for k in xrange(svec_mat.shape[0]):
            # shift left to make room
            svec[si] = svec[si] << 1
            # set bit if we want to
            if svec_mat[k,si]:
                svec[si] = svec[si] ^ 0x01
    # Loop over temperatures
    for itemp in xrange(schedsize):
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
                        # loop over bits
                        for k in xrange(64):
                            # zero maps to one, one maps to negative 1
                            if not getbit(svec[sidx], 63 - k):
                                ediffs[k] -= 2.0*jval
                            else:
                                ediffs[k] += 2.0*jval
                    # quadratic part
                    else:
                        # do the XOR to see bit disagreements
                        flipmask = svec[sidx] ^ svec[spinidx]
                        # loop over bits
                        for k in xrange(64):
                            # zero maps to one, one maps to negative 1
                            if not getbit(flipmask, 63 - k):
                                ediffs[k] -= 2.0*jval
                            else:
                                ediffs[k] += 2.0*jval
                # prepare to flip those whose Boltzmann weights are 
                # larger than random samples
                sign = np.asarray(np.exp(ediffs/temp) > rands, dtype=np.int8)
                # set a one and shift left
                flipmask = 1 if sign[0] else 0
                # go through all except the first
                for k in xrange(1, 64):
                    # shift last value left to make room
                    flipmask = flipmask << 1
                    # if we want to flip, set a one
                    if sign[k]:
                        flipmask ^= 0x01
                # do the flip
                svec[sidx] ^= flipmask
                # reset energy differences
                for k in xrange(64):
                    ediffs[k] = 0.0
                # new random numbers
                rands = rng.rand(64)
            # reshuffle update order
            sidx_shuff = rng.permutation(sidx_shuff)
    # unpack and return
    for sidx in xrange(nspins+1):
        state = bin(svec[sidx])[2:].rjust(64,'0')
        for k in xrange(len(state)):
            svec_mat[k,sidx] = float(state[k])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal_bipartite(np.float_t[:] sched, 
                       int mcsteps, 
                       np.float_t[:] lvec, 
                       np.float_t[:] rvec, 
                       np.float_t[:, :] J, 
                       np.float_t[:] lbias,
                       np.float_t[:] rbias,
                       rng):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is 
    given by @lvec and @rvec, which we update in-place and calculate 
    energies using the "neighbors array" @nbs.

    This version is for a bipartite Ising graph whose state is defined
    using @lvec and @rvec, a vector of spins for the "left" and "right"
    spins. We update and calculate energies using the Ising graph @J,
    which has shape (@lvec.size, @rvec.size). @lbias and @rbias are 
    vectors of size @lvec and @rvec, which encode the biases on these
    spins.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @lvec (np.array, float): contains the starting configuration for
                                 the "left" collection of independent
                                 spins
        @rvec (np.array, float): contains the starting configuration for
                                 the "right" collection of independent
                                 spins
        @lbias (np.array, float): contains the "left" collection local
                                  field biases
        @rbias (np.array, float): contains the "right" collection local
                                  field biases
        @J (np.ndarray, float): 2D matrix that stores coupling values on
                                the off-diagonals, and nothing on the 
                                diagonals
        @rng (np.RandomState): numpy random number generator object

    Returns:
        None: spins are flipped in-place within @svec
    """
    # Define some variables
    cdef int lspins = lvec.size
    cdef int rspins = rvec.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef float ediff = 0.0
    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over left spins
            for sidx in xrange(lspins):
                # loop through the given spin's neighbors
                for si in xrange(rspins):
                    ediff -= 2.0*lvec[sidx]*J[sidx,si]*rvec[si] + \
                        lbias[sidx]*lvec[sidx] + \
                        rbias[si]*rvec[si]
                # Metropolis accept or reject
                if ediff > 0.0:  # avoid overflow
                    lvec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    lvec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            # Loop over right spins
            for sidx in xrange(rspins):
                # loop through the given spin's neighbors
                for si in xrange(lspins):
                    ediff -= 2.0*rvec[sidx]*J[si,sidx]*lvec[si] + \
                        lbias[si]*lvec[si] + \
                        rbias[sidx]*rvec[sidx]
                # Metropolis accept or reject
                if ediff > 0.0:  # avoid overflow
                    rvec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    rvec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal_cuda(np.ndarray[np.float_t, ndim=1] sched, 
                  int mcsteps, 
                  np.ndarray[np.float_t, ndim=1] svec, 
                  np.ndarray[np.float_t, ndim=3] nbs, 
                  rng):
    """
    Execute thermal annealing according to @sched with @mcsteps
    sweeps for each annealing step. Starting configuration is 
    given by @svec, which we update in-place and calculate energies
    using the "neighbors array" @nbs.

    This version uses PyCUDA and an Nvidia GPU to parallelize over 
    inner spin-update loop.

    Args:
        @sched (np.array, float): an array of temperatures that specify
                                  the annealing schedule
        @mcsteps (int): number of sweeps to do on each annealing step
        @svec (np.array, float): contains the starting configuration
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
    # get what we need for pycuda
    import sys
    from pycuda import driver, compiler, gpuarray, tools, characterize
    from pycuda.curandom import rand as curand
    # initialize device
    import pycuda.autoinit
    # Define some variables
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int schedsize = sched.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    # kernel
    cuda_kernel_template = """
#include <math.h>
#include <curand_kernel.h>

extern "C" {
__global__ void SetupRNGs(curandState *state, unsigned long seed)
{
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}
__global__ void UpdateSpins(double temp, double *nbs, double *spins, curandState *gstate)
{
    int ni, nidx;
    double jval;

    double ediff = 0.0;
    const uint sidx = threadIdx.x;

    // loop over the neighbors of this spin
    for(ni=0; ni < %(MAXNB)s; ni++)
    {
        // get the neighbor spin index
        nidx = (int) nbs[sidx * %(MAXNB)s * 2 + ni*2];
        // get coupling value to that neighbor
        jval = nbs[sidx * %(MAXNB)s * 2 + ni*2 + 1];
        // self-connections are not quadratic
        if(nidx == sidx)
            ediff += -2.0*spins[sidx]*jval;
        else
            ediff += -2.0*spins[sidx]*(jval*spins[nidx]);
    }
    // Metropolis -- accept or reject
    if(ediff >= 0.0)
        spins[sidx] *= -1;
    else if(exp(ediff/temp) > curand_uniform(&gstate[sidx]))
        spins[sidx] *= -1;
}
}
"""
    cuda_kernel = cuda_kernel_template % {'NSPINS': nspins, 'MAXNB': maxnb}
    cumodule = compiler.SourceModule(cuda_kernel, no_extern_c=True, keep=True)
    cuSetupRNGs = cumodule.get_function("SetupRNGs")
    cuUpdateSpins = cumodule.get_function("UpdateSpins")
    # setup the random number generators
    rngstate = driver.mem_alloc(nspins * 
                                characterize.sizeof('curandStateXORWOW', 
                                                    '#include <curand_kernel.h>'))
    cuSetupRNGs(rngstate, np.int16(0), block=(nspins,1,1))
    # transfer from host to device
    svec_g = gpuarray.to_gpu(svec)
    # sched_g = gpuarray.to_gpu(sched)
    nbs_g = gpuarray.to_gpu(nbs)
    # Loop over temperatures
    for itemp in xrange(schedsize):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            cuUpdateSpins(np.float64(temp), nbs_g, svec_g, rngstate,
                          block=(nspins,1,1))
            print svec_g.get()


            # # Loop over spins
            # for sidx in xrange(nspins):
            #     # loop through the given spin's neighbors
            #     for si in xrange(maxnb):
            #         # get the neighbor spin index
            #         spinidx = int(nbs[sidx,si,0])
            #         # get the coupling value to that neighbor
            #         jval = nbs[sidx,si,1]
            #         # self-connections are not quadratic
            #         if spinidx == sidx:
            #             ediff += -2.0*svec[sidx]*jval
            #         # calculate the energy diff of flipping this spin
            #         else:
            #             ediff += -2.0*svec[sidx]*(jval*svec[spinidx])
            #     # Metropolis accept or reject
            #     if ediff >= 0.0:  # avoid overflow
            #         svec[sidx] *= -1
            #     elif cexp(ediff/temp) > crand()/float(RAND_MAX):
            #         svec[sidx] *= -1
            #     # Reset energy diff value
            #     ediff = 0.0

    # transfer from device back to host
    svec[:] = svec_g.get()

