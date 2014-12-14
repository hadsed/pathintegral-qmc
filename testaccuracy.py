'''

File:
Author: Hadayat Seddiqi
Date: 12.11.14
Description: Test the accuracy of PIQA code by comparing with 
             a presolved Ising instance.

'''

import pickle
import numpy as np
import scipy.sparse as sps
import piqa
import sa

# Define some parameters
nrows = 32
fieldstart = 1.5
fieldend = 1e-8
preannealing = True
preannealingsteps = 100
preannealingtemp = 3.0
seed = None
annealingtemp = 0.01
trotterslices = 20
annealingsteps = 100
# Random number generator
rng = np.random.RandomState(seed)
# Test file name
inputfname = 'ising_instances/inst_0.npz'

# Construct the ground state spin configuration from spinglass server results
groundstate = -np.ones(32*32)
gsspinups = np.array([2,4,5,6,7,10,11,12,13,14,15,17,19,20,25,26,32,35,36,39,40,41,44,46,47,52,53,54,56,59,66,67,69,70,72,74,76,79,84,88,89,91,92,94,96,97,98,101,102,103,104,106,108,112,114,115,119,122,123,125,126,127,129,130,133,134,136,137,138,139,140,142,146,149,150,151,154,156,159,161,162,163,166,169,172,177,179,180,181,183,186,188,190,192,194,196,197,201,204,205,210,213,215,216,217,219,220,224,228,229,232,237,238,239,241,244,245,246,248,251,252,253,255,257,260,261,262,263,265,267,269,270,271,272,273,274,276,277,279,281,283,284,285,288,292,294,295,296,298,300,301,303,305,307,308,311,312,313,319,321,322,323,326,329,331,336,338,340,342,348,349,350,353,355,357,360,361,362,365,368,373,375,376,377,378,384,386,387,388,389,392,393,394,396,398,399,400,401,404,405,406,408,410,412,415,417,418,419,421,423,425,429,432,435,436,439,441,442,446,448,449,450,457,462,463,464,465,466,469,470,471,472,473,474,475,476,481,484,487,492,493,495,496,497,499,505,507,508,510,511,512,514,517,520,522,524,526,527,535,540,541,544,546,547,548,549,550,551,552,555,559,562,564,566,567,568,576,578,579,580,581,582,583,587,589,590,591,592,599,600,602,605,607,609,611,615,616,617,619,621,622,623,624,626,630,631,632,633,634,635,637,638,639,642,643,645,647,651,653,654,655,656,657,658,660,662,663,664,667,671,673,677,678,679,680,683,684,685,686,689,691,692,693,697,698,699,700,701,702,703,704,705,707,708,710,711,712,713,714,716,717,718,722,724,727,728,730,732,736,741,742,744,746,749,752,753,760,762,765,766,769,770,771,773,775,782,786,793,798,802,804,806,809,811,812,813,815,816,820,822,825,826,827,829,832,833,836,842,843,844,846,848,850,851,853,854,857,862,866,868,869,870,871,872,873,874,875,876,878,880,881,882,885,887,890,891,893,894,896,897,898,899,901,904,906,908,909,910,911,916,919,920,923,928,932,933,934,936,938,939,940,944,947,950,951,954,955,958,959,960,963,964,968,970,972,973,976,977,984,985,986,988,990,991,993,994,995,996,997,999,1001,1004,1006,1007,1009,1010,1011,1013,1014,1016,1017,1019,1020,1021,1022,1024]) - 1
# flip the spins that shouldn't be up
groundstate[gsspinups] = 1
# Read in the diagonals of the 2D Ising instance
loader = np.load(inputfname)
nSpins = loader['nSpins'][0]
# Reconstruct the matrix in sparse diagonal format
isingJ = sps.dia_matrix(([loader['hcons'], loader['vcons'],
                          loader['phcons'], loader['pvcons']],
                         loader['k']),
                        shape=(nSpins, nSpins))

# calculate the actual ground state energy
gsenergy = np.dot(groundstate, -isingJ.dot(groundstate))
print "True groundstate energy: ", gsenergy

# calculate ground state energy using SA
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nSpins) ], 
                      dtype=np.float)
print ("Initial SA energy: ", sa.ClassicalIsingEnergy(spinVector, isingJ))
# Do the pre-annealing
sa.Anneal(preannealingtemp, annealingtemp,
          preannealingsteps, spinVector, isingJ, rng)
print ("Final SA energy: ", sa.ClassicalIsingEnergy(spinVector, isingJ))
print str(np.sum(spinVector == groundstate))+'/1024 spins agree'

# Now see what PIQA gives us
minenergy, minconfig = piqa.SimulateQuantumAnnealing(
    trotterslices, nrows, annealingtemp, annealingsteps, 
    fieldstart, fieldend, preannealing, preannealingsteps, 
    preannealingtemp, seed, inputfname, False
    )
print "PIQA reported energy: ", minenergy
print str(np.sum(minconfig == groundstate))+'/1024 spins agree'
