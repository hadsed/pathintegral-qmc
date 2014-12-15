'''

File:
Author: Hadayat Seddiqi
Date: 12.11.14
Description: Test the accuracy of PIQA code by comparing with 
             a presolved Ising instance.

'''

import time
import pickle
import numpy as np
import scipy.sparse as sps
import piqa

# Define some parameters
nrows = 32
preannealing = True
preannealingsteps = 100
preannealingtemp = 3.0
seed = None
annealingtemp = 0.01
trotterslices = 20
annealingsteps = 100
fieldstart = 1.5
fieldend = 1e-8
fieldstep = ((fieldstart-fieldend)/annealingsteps)
# Random number generator
rng = np.random.RandomState(seed)
# Test file name
inputfname = 'ising_instances/inst_0_32x32'

# Construct the ground state spin configuration from spinglass server results
groundstate = -np.ones(32*32)
gsspinups = np.array([1,2,4,5,6,9,10,11,12,13,16,22,23,25,27,28,30,31,32,34,37,39,42,44,45,47,48,50,51,56,59,61,62,63,66,68,69,71,74,75,77,78,80,81,84,86,88,90,92,93,96,98,100,102,103,104,108,109,110,114,115,116,117,119,120,121,125,128,130,131,133,134,135,137,138,139,141,143,146,147,155,158,159,160,161,162,164,165,169,171,172,173,174,176,177,179,181,182,187,188,192,193,194,198,200,201,204,206,207,208,210,214,217,221,222,224,228,229,230,233,234,235,237,239,240,241,242,243,245,249,256,260,265,267,270,271,272,276,278,282,283,286,288,289,292,297,302,303,304,306,308,310,311,313,314,315,318,320,321,323,325,326,328,330,331,332,334,336,341,342,343,345,347,348,349,352,354,356,357,358,359,362,363,364,366,368,370,372,374,375,377,378,379,384,388,389,390,392,394,395,401,402,404,405,407,410,411,412,414,417,419,420,427,428,433,434,435,438,439,440,445,446,447,448,449,451,457,458,464,468,471,474,476,479,481,482,483,484,485,487,489,491,493,497,499,502,503,505,506,507,509,510,511,515,516,517,519,520,523,524,526,528,534,535,536,537,538,539,542,545,546,548,550,557,559,561,562,563,564,567,572,573,575,577,579,581,584,585,588,591,592,593,594,599,601,606,608,611,613,616,618,619,620,622,623,630,636,637,640,642,644,645,646,647,648,649,651,653,655,656,657,659,660,661,663,668,669,670,671,676,679,680,684,685,688,690,691,692,693,697,699,700,701,703,708,711,712,714,717,718,728,729,731,735,737,738,740,745,747,748,750,751,753,754,756,757,760,761,762,766,768,771,774,776,777,781,787,788,791,796,798,799,803,805,806,807,808,809,810,812,815,819,823,824,830,832,833,839,840,842,846,847,848,849,850,852,854,859,863,869,870,875,876,878,879,881,883,884,885,886,891,892,893,895,897,898,901,905,906,908,915,917,921,922,924,925,926,927,933,934,935,937,938,939,940,941,942,949,951,952,954,956,957,958,959,964,974,975,977,979,980,987,989,995,997,1001,1002,1003,1004,1008,1012,1013,1018,1019,1021,1022])-1
# this one is for inst_0.txt, not inst_0_32x32.txt
# gsspinups = np.array([2,4,5,6,7,10,11,12,13,14,15,17,19,20,25,26,32,35,36,39,40,41,44,46,47,52,53,54,56,59,66,67,69,70,72,74,76,79,84,88,89,91,92,94,96,97,98,101,102,103,104,106,108,112,114,115,119,122,123,125,126,127,129,130,133,134,136,137,138,139,140,142,146,149,150,151,154,156,159,161,162,163,166,169,172,177,179,180,181,183,186,188,190,192,194,196,197,201,204,205,210,213,215,216,217,219,220,224,228,229,232,237,238,239,241,244,245,246,248,251,252,253,255,257,260,261,262,263,265,267,269,270,271,272,273,274,276,277,279,281,283,284,285,288,292,294,295,296,298,300,301,303,305,307,308,311,312,313,319,321,322,323,326,329,331,336,338,340,342,348,349,350,353,355,357,360,361,362,365,368,373,375,376,377,378,384,386,387,388,389,392,393,394,396,398,399,400,401,404,405,406,408,410,412,415,417,418,419,421,423,425,429,432,435,436,439,441,442,446,448,449,450,457,462,463,464,465,466,469,470,471,472,473,474,475,476,481,484,487,492,493,495,496,497,499,505,507,508,510,511,512,514,517,520,522,524,526,527,535,540,541,544,546,547,548,549,550,551,552,555,559,562,564,566,567,568,576,578,579,580,581,582,583,587,589,590,591,592,599,600,602,605,607,609,611,615,616,617,619,621,622,623,624,626,630,631,632,633,634,635,637,638,639,642,643,645,647,651,653,654,655,656,657,658,660,662,663,664,667,671,673,677,678,679,680,683,684,685,686,689,691,692,693,697,698,699,700,701,702,703,704,705,707,708,710,711,712,713,714,716,717,718,722,724,727,728,730,732,736,741,742,744,746,749,752,753,760,762,765,766,769,770,771,773,775,782,786,793,798,802,804,806,809,811,812,813,815,816,820,822,825,826,827,829,832,833,836,842,843,844,846,848,850,851,853,854,857,862,866,868,869,870,871,872,873,874,875,876,878,880,881,882,885,887,890,891,893,894,896,897,898,899,901,904,906,908,909,910,911,916,919,920,923,928,932,933,934,936,938,939,940,944,947,950,951,954,955,958,959,960,963,964,968,970,972,973,976,977,984,985,986,988,990,991,993,994,995,996,997,999,1001,1004,1006,1007,1009,1010,1011,1013,1014,1016,1017,1019,1020,1021,1022,1024]) - 1
# flip the spins that shouldn't be up
groundstate[gsspinups] = 1
nSpins = nrows**2
# Read from textfile directly to be sure
loaded = np.loadtxt(inputfname+'.txt')
isingJ = sps.dok_matrix((nSpins,nSpins))
for i,j,val in loaded:
    isingJ[i-1,j-1] = val
# calculate the actual ground state energy
gsenergy = np.dot(groundstate, -isingJ.dot(groundstate))
print "True groundstate energy: ", gsenergy
print "True energy per spin: ", gsenergy/float(nSpins)
print "True magnetization: ", np.sum(groundstate)
print "True magnetization per spin: ", np.sum(groundstate)/float(nSpins)
print '\n'
# calculate ground state energy using SA
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nSpins) ], 
                      dtype=np.float)
spinVector2 = spinVector.copy()
print ("Initial state energy: ", piqa.sa.ClassicalIsingEnergy(spinVector, isingJ))
print '\n'

# optimized SA
t2 = time.time()
piqa.sa.Anneal_opt(preannealingtemp, annealingtemp,
                   preannealingsteps, spinVector2, isingJ, rng)
t3 = time.time()
print ("Final SA_opt energy: ", piqa.sa.ClassicalIsingEnergy(spinVector2, isingJ))
print "SA_opt time: ", str(t3-t2)
print '\n'
# non-opt SA
isingJ = isingJ.todia()
t0 = time.time()
piqa.sa.Anneal(preannealingtemp, annealingtemp,
               preannealingsteps, spinVector, isingJ, rng)
t1 = time.time()
print ("Final SA energy: ", piqa.sa.ClassicalIsingEnergy(spinVector, isingJ))
print str(np.sum(spinVector == groundstate))+'/1024 spins agree'
print "SA time: ", str(t1-t0)
print '\n'

os.wearedonerightnow


# Now see what PIQA gives us
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nSpins) ], 
                      dtype=np.float)
configurations = np.tile(spinVector, (trotterslices, 1)).T
isingJ = isingJ.tocsr()
perpJ = sps.dia_matrix(([[-trotterslices*annealingtemp/2.], 
                         [-trotterslices*annealingtemp/2.]], 
                        [1, trotterslices-1]), 
                       shape=(trotterslices, trotterslices))
piqa.sa.Anneal(preannealingtemp, annealingtemp,
               1, spinVector, isingJ, rng)
piqa.qmc.QuantumAnneal(fieldstart, fieldstep, annealingsteps, 
                       trotterslices, annealingtemp, nSpins, 
                       perpJ, isingJ, configurations, rng)
minEnergy, minConfiguration = np.inf, []
for col in configurations.T:
    candidateEnergy = piqa.sa.ClassicalIsingEnergy(col, isingJ)
    if candidateEnergy < minEnergy:
        minEnergy = candidateEnergy
        minConfiguration = col
print "PIQA reported energy: ", minEnergy
print str(np.sum(minConfiguration == groundstate))+'/1024 spins agree'
