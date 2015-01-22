'''

File: spinglass32_mpi.py
Author: Hadayat Seddiqi
Date: 01.22.15
Description: Do the 32x32 square lattice Ising system
             using MPI to run several simulation instances
             at once. The problem is the same as the non-MPI
             version, with the same answer.

'''

import numpy as np
import scipy.sparse as sps

import piqmc.sa as sa
import piqmc.qmc as qmc
import piqmc.tools as tools

from mpi4py import MPI

comm = MPI.COMM_WORLD

# Number of runs per processor
nruns = 20
nproc = comm.size

# Define some parameters
nrows = 32
nspins = nrows**2
preannealingsteps = 10
preannealingmcsteps = 1
preannealingtemp = 3.0
annealingtemp = 0.01
annealingsteps = 100
annealingmcsteps = 1
fieldstart = 1.5
fieldend = 1e-8
trotterslices = 20
seed = None
# Random number generator
rng = np.random.RandomState(seed)
# Test file name
inputfname = 'ising_instances/inst_0_32x32'

# Construct the ground state spin configuration from spinglass server results
groundstate = -np.ones(32*32)
gsspinups = np.array([1,2,4,5,6,9,10,11,12,13,16,22,23,25,27,28,30,31,32,34,37,39,42,44,45,47,48,50,51,56,59,61,62,63,66,68,69,71,74,75,77,78,80,81,84,86,88,90,92,93,96,98,100,102,103,104,108,109,110,114,115,116,117,119,120,121,125,128,130,131,133,134,135,137,138,139,141,143,146,147,155,158,159,160,161,162,164,165,169,171,172,173,174,176,177,179,181,182,187,188,192,193,194,198,200,201,204,206,207,208,210,214,217,221,222,224,228,229,230,233,234,235,237,239,240,241,242,243,245,249,256,260,265,267,270,271,272,276,278,282,283,286,288,289,292,297,302,303,304,306,308,310,311,313,314,315,318,320,321,323,325,326,328,330,331,332,334,336,341,342,343,345,347,348,349,352,354,356,357,358,359,362,363,364,366,368,370,372,374,375,377,378,379,384,388,389,390,392,394,395,401,402,404,405,407,410,411,412,414,417,419,420,427,428,433,434,435,438,439,440,445,446,447,448,449,451,457,458,464,468,471,474,476,479,481,482,483,484,485,487,489,491,493,497,499,502,503,505,506,507,509,510,511,515,516,517,519,520,523,524,526,528,534,535,536,537,538,539,542,545,546,548,550,557,559,561,562,563,564,567,572,573,575,577,579,581,584,585,588,591,592,593,594,599,601,606,608,611,613,616,618,619,620,622,623,630,636,637,640,642,644,645,646,647,648,649,651,653,655,656,657,659,660,661,663,668,669,670,671,676,679,680,684,685,688,690,691,692,693,697,699,700,701,703,708,711,712,714,717,718,728,729,731,735,737,738,740,745,747,748,750,751,753,754,756,757,760,761,762,766,768,771,774,776,777,781,787,788,791,796,798,799,803,805,806,807,808,809,810,812,815,819,823,824,830,832,833,839,840,842,846,847,848,849,850,852,854,859,863,869,870,875,876,878,879,881,883,884,885,886,891,892,893,895,897,898,901,905,906,908,915,917,921,922,924,925,926,927,933,934,935,937,938,939,940,941,942,949,951,952,954,956,957,958,959,964,974,975,977,979,980,987,989,995,997,1001,1002,1003,1004,1008,1012,1013,1018,1019,1021,1022])-1
# flip the spins that shouldn't be up
groundstate[gsspinups] = 1
# Read from textfile directly to be sure
loaded = np.loadtxt(inputfname+'.txt')
isingJ = sps.dok_matrix((nspins,nspins))
for i,j,val in loaded:
    isingJ[i-1,j-1] = val
# calculate the actual ground state energy
gsenergy = np.dot(groundstate, -isingJ.dot(groundstate))
# print "True groundstate energy: ", gsenergy
# print "True energy per spin: ", gsenergy/float(nspins)
# print "True magnetization: ", np.sum(groundstate)
# print "True magnetization per spin: ", np.sum(groundstate)/float(nspins)
# print '\n'

# calculate ground state energy
# configurations = np.tile(spinVector, (trotterslices, 1)).T

# Generate list of nearest-neighbors for each spin
neighbors = tools.GenerateNeighbors(nspins, isingJ, 4)

# Try just using SA
tannealingsched = np.linspace(preannealingtemp,
                              annealingtemp,
                              annealingsteps)
for i in xrange(nruns):
    spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                          dtype=np.float)
    # print (comm.rank, "Initial state energy: ", sa.ClassicalIsingEnergy(spinVector, isingJ))
    sa.Anneal(tannealingsched, annealingmcsteps, spinVector, neighbors, rng)
    print (comm.rank, "Residual energy: ", 
           sa.ClassicalIsingEnergy(spinVector, isingJ) - gsenergy)


# # Now do PIQMC
# preannealingsched = np.linspace(preannealingtemp, 
#                                 annealingtemp, 
#                                 100)
# annealingsched = np.linspace(fieldstart,
#                              fieldend,
#                              annealingsteps)
# t0 = time.time()
# sa.Anneal(preannealingsched, 1, spinVector, neighbors, rng)  # preannealing
# qmc.QuantumAnneal(annealingsched, annealingmcsteps,
#                   trotterslices, annealingtemp, nspins, 
#                   configurations, neighbors, rng)
# t1 = time.time()
# print "PIQMC time (seconds): ", str(t1-t0)
# minEnergy, minConfiguration = np.inf, []
# for col in configurations.T:
#     candidateEnergy = sa.ClassicalIsingEnergy(col, isingJ)
#     if candidateEnergy < minEnergy:
#         minEnergy = candidateEnergy
#         minConfiguration = col
# print "PIQMC energy: ", minEnergy
# print str(np.sum(minConfiguration == groundstate))+'/1024 spins agree'
