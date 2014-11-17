'''

File: qmc.py
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do the path-integral quantum annealing.

'''

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def QuantumIsingEnergy(np.ndarray[DTYPE_t, ndim=1] spins, 
                       np.ndarray[DTYPE_t, ndim=1] tspins, 
                       J, Jperp):
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

def QuantumMetropolisAccept(rng, 
                            np.ndarray[DTYPE_t, ndim=1] svec, 
                            int fidx, 
                            np.ndarray[DTYPE_t, ndim=1]tvec, 
                            J, Jperp, 
                            float T):
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

cdef inline double dot(double[:] v1, 
                       double[:] v2, 
                       int length) nogil:
    cdef double result = 0
    cdef int i = 0
    # cdef int length = v1.size
    cdef double el1 = 0
    cdef double el2 = 0
    for i in range(length):
        el1 = v1[i]
        el2 = v2[i]
        result += el1*el2
    return result

def QuantumAnneal(float transFieldStart, float transFieldStep, 
    		  int annealingSteps, int trotterSlices, 
		  float annealingTemperature, int nSpins, perpJ, 
		  isingJ, 
                  np.ndarray[DTYPE_t, ndim=2] configurations, 
                  rng):
    """
    Execute quantum annealing part using path-integral quantum Monte Carlo.
    The quantum annealing is controlled by the transverse field which starts
    at @transFieldStart and decreases by @transFieldStep for @annealingSteps
    number of steps. The ambient temperature is @annealingTemperature, and the
    total number of spins is @nSpins. @isingJ and @perpJ give the parts of the
    Hamiltonian to calculate the energies, and @configurations is a list of
    spin vectors of length @trotterSlices. @rng is the random number generator.

    Returns: None (spins are flipped in-place)
    """
    cdef int accept = 0
    cdef int ispin = 0
    cdef int ifield = 0
    cdef int islice = 0
    cdef int slices = trotterSlices
    cdef float Ej1 = 0
    cdef double[:,:] conf_view = configurations

    cdef np.ndarray[DTYPE_t, ndim=1] E0 = np.zeros(nSpins)
    cdef np.ndarray[DTYPE_t, ndim=1] E1 = np.zeros(nSpins)
    cdef np.ndarray[DTYPE_t, ndim=1] randomUniformSamples

    randomUniformSamples = rng.uniform(0,1, nSpins*annealingSteps*trotterSlices)

    # Loop over transverse field annealing schedule
    for ifield, field in enumerate((transFieldStart - k*transFieldStep
                                    for k in xrange(annealingSteps+1))):
	# Calculate new coefficient for 1D Ising J
        perpJCoeff = np.log(np.tanh(field / 
                                    (trotterSlices*annealingTemperature)))
        calculatedPerpJ = perpJCoeff*perpJ
        # Loop over Trotter slices
        for islice in rng.permutation(range(trotterSlices)):
        # for islice in prange(trotterSlices, nogil=True):
            # Loop over spins
            # for ispin in rng.permutation(range(nSpins)):
            print E0
            for ispin in prange(nSpins, nogil=True):
                # Grab nearest-neighbor spin vector across Trotter slices
                # trotterSpins = configurations[ispin, :]
                # Attempt to flip this spin
                # if QuantumMetropolisAccept(rng, configurations[:, islice], 
                #                            ispin, 
                #                            configurations[ispin, :], ###
                #                            isingJ, 
                #                            calculatedPerpJ, 
                #                            annealingTemperature):
                #     configurations[ispin, islice] *= -1

                # e0 = QuantumIsingEnergy(svec, tvec, J, Jperp)
                # svec[fidx] *= -1
                # e1 = QuantumIsingEnergy(svec, tvec, J, Jperp)
                # svec[fidx] *= -1  # we're dealing with the original array, so flip back
                E0[ispin] = -1.0*slices
                E0[ispin] = dot(conf_view[islice], 
                                conf_view[islice],
                                nSpins)
                # E0[ispin] = dot(configurations[islice], 
                #                 configurations[islice],
                #                 nSpins)
                # E0[ispin] = -1.0*slices*(np.dot(configurations[:][islice], 
                #                              isingJ.dot(configurations[:][islice])) + 
                #                       np.dot(configurations[ispin], 
                #                              calculatedPerpJ.dot(configurations[ispin])))

                # if (e0 - e1) > 0.0:  # avoid overflow
                #     accept = 1
                # if np.exp((e0 - e1)/T) > randomUniformSamples[ifield+islice+ispin]:
                #     accept = 1
                # else:
                #     accept = 0
            print E0
