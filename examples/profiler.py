#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py
'''

File: profiler.py
Author: Hadayat Seddiqi
Date: 01.15.15
Description: Profile the simulated and quantum annealing routines.

'''

import pstats, cProfile
import scipy.sparse as sps
import numpy as np

import piqmc.sa as sa
import piqmc.qmc as qmc
import piqmc.tools as tools


# Define some parameters
nrows = 80
nspins = nrows**2
preannealingsteps = 25
preannealingmcsteps = 1
preannealingtemp = 3.0
annealingtemp = 0.01
annealingsteps = 25
annealingmcsteps = 1
fieldstart = 1.5
fieldend = 1e-8
seed = None
trotterslices = 5
# Random number generator
rng = np.random.RandomState(seed)

# Read from textfile directly to be sure
inputfname = 'ising_instances/santoro_80x80.txt'
loaded = np.loadtxt(inputfname)
isingJ = sps.dok_matrix((nspins,nspins))
for i,j,val in loaded:
    isingJ[i-1,j-1] = val

# Get list of nearest-neighbors for each spin
neighbors = tools.GenerateNeighbors(
    nspins, 
    isingJ, 
    4,
    'ising_instances/santoro_80x80_neighbors.npy'
    )
# neighbors = np.load('ising_instances/santoro_80x80_neighbors.npy')

# initialize the state
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                      dtype=np.float)
configurations = np.tile(spinVector, (trotterslices, 1)).T
# Try just using SA
tannealingsched = np.linspace(preannealingtemp,
                              annealingtemp,
                              annealingsteps)
cProfile.runctx("sa.Anneal(tannealingsched, annealingmcsteps, "
                "spinVector, neighbors, rng)",
                globals(), locals(), "sa.prof")
s = pstats.Stats("sa.prof")
s.strip_dirs().sort_stats("time").print_stats()
# Now do PIQA
annealingsched = np.linspace(fieldstart,
                             fieldend,
                             annealingsteps)
cProfile.runctx("qmc.QuantumAnneal(annealingsched, annealingmcsteps,"
                "trotterslices, annealingtemp, nspins, "
                "configurations, neighbors, rng)",
                globals(), locals(), "qmc.prof")
s = pstats.Stats("qmc.prof")
s.strip_dirs().sort_stats("time").print_stats()
