'''

File: qmc.py
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do the path-integral quantum annealing.

'''

# cimport cython
# from cython.parallel import prange
import numpy as np
cimport numpy as np


def QuantumIsingEnergy(np.ndarray[np.float_t, ndim=1] spins, 
                       np.ndarray[np.float_t, ndim=1] tspins, 
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

def QuantumMetropolisAccept(rng, np.ndarray[np.float_t, ndim=1] svec, 
                            int fidx, np.ndarray[np.float_t, ndim=1] tvec,
                            J, Jperp, float T):
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

def QuantumAnneal(float transFieldStart, float transFieldStep, 
                  int annealingSteps, int trotterSlices, 
		  float annealingTemperature, int nSpins, 
                  perpJ, isingJ, 
                  np.ndarray[np.float_t, ndim=2] configurations, 
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
    # Loop over transverse field annealing schedule
    for ifield, field in enumerate((transFieldStart - k*transFieldStep
                                    for k in xrange(annealingSteps+1))):
	# Calculate new coefficient for 1D Ising J
        perpJCoeff = np.log(np.tanh(field / 
                                    (trotterSlices*annealingTemperature)))
        calculatedPerpJ = perpJCoeff*perpJ
        # Loop over Trotter slices
        for islice in rng.permutation(range(trotterSlices)):
            # Loop over spins
            for ispin in rng.permutation(range(nSpins)):
                # Attempt to flip this spin
                if QuantumMetropolisAccept(rng, 
                                           configurations[:, islice], 
                                           ispin, 
                                           configurations[ispin, :],
                                           isingJ, 
                                           calculatedPerpJ, 
                                           annealingTemperature):
                    configurations[ispin, islice] *= -1
