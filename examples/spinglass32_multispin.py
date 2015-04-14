'''

File: spinglass32_multispin.py
Author: Hadayat Seddiqi
Date: 02.26.15
Description: Test the performance of the Anneal_multispin method
             for SA by comparing with a presolved Ising instance
             from M. Juenger's spin glass server. Here we're using
             inst_0_32x32.txt. Spin glass server returned the 
             following email:


The Spin Glass Server (SGS) completed your Metajob:

        Name: inst_0_32x32.txt
        Output Type: long
        Containing Jobs: 1
        Internal Metajob ID: 693408
        Date: Mon Dec 15 21:48:18 2014

result:

-------------------------- JOB: 693407 ----------------------------------

Name: /tmp/haYuHoDcDL/0JpnZJTelT
Problemtype: gauss
Size: 32

Machine: Intel Core i7-3770T @ 2.50GHz-3.70GHz

result:

        Energy (per spin): -1.55460615142578
        Magnetization (per spin):0.0703125000000
        CPU-Time (min:sec.1/100): 0:01.06

Here comes an optimal spin configuration:

        total number of spins: 1024
        spins up: 476

CONFIGURATION_BEGIN
 1 2 4 5 6 9 10 11 12 13 16 22 23 25 27 28 30 31 32 34 37 39 42 44 45 47 48 50 51 56 59 61 62 63 66 68 69 71 74 75 77 78 80 81 84 86 88 90 92 93 96 98 100 102 103 104 108 109 110 114 115 116 117 119 120 121 125 128 130 131 133 134 135 137 138 139 141 143 146 147 155 158 159 160 161 162 164 165 169 171 172 173 174 176 177 179 181 182 187 188 192 193 194 198 200 201 204 206 207 208 210 214 217 221 222 224 228 229 230 233 234 235 237 239 240 241 242 243 245 249 256 260 265 267 270 271 272 276 278 282 283 286 288 289 292 297 302 303 304 306 308 310 311 313 314 315 318 320 321 323 325 326 328 330 331 332 334 336 341 342 343 345 347 348 349 352 354 356 357 358 359 362 363 364 366 368 370 372 374 375 377 378 379 384 388 389 390 392 394 395 401 402 404 405 407 410 411 412 414 417 419 420 427 428 433 434 435 438 439 440 445 446 447 448 449 451 457 458 464 468 471 474 476 479 481 482 483 484 485 487
 489 491 493 497 499 502 503 505 506 507 509 510 511 515 516 517 519 520 523 524 526 528 534 535 536 537 538 539 542 545 546 548 550 557 559 561 562 563 564 567 572 573 575 577 579 581 584 585 588 591 592 593 594 599 601 606 608 611 613 616 618 619 620 622 623 630 636 637 640 642 644 645 646 647 648 649 651 653 655 656 657 659 660 661 663 668 669 670 671 676 679 680 684 685 688 690 691 692 693 697 699 700 701 703 708 711 712 714 717 718 728 729 731 735 737 738 740 745 747 748 750 751 753 754 756 757 760 761 762 766 768 771 774 776 777 781 787 788 791 796 798 799 803 805 806 807 808 809 810 812 815 819 823 824 830 832 833 839 840 842 846 847 848 849 850 852 854 859 863 869 870 875 876 878 879 881 883 884 885 886 891 892 893 895 897 898 901 905 906 908 915 917 921 922 924 925 926 927 933 934 935 937 938 939 940 941 942 949 951 952 954 956 957 958 959 964 974 975 977 979 980 987 989 995 997 1001 1002 1003 1004 1008 1012 1013 1018 1019 1021 1022
CONFIGURATION_END

'''

import time
import numpy as np
import scipy.sparse as sps

import piqmc.sa as sa
import piqmc.tools as tools


# Define some parameters
nrows = 32
nspins = nrows**2
preannealingtemp = 3.0
annealingtemp = 0.01
annealingsteps = 10000
annealingmcsteps = 1
seed = 1234
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
print "True groundstate energy: ", gsenergy
print "True energy per spin: ", gsenergy/float(nspins)
print "True magnetization: ", np.sum(groundstate)
print "True magnetization per spin: ", np.sum(groundstate)/float(nspins)
print '\n'

# make a bunch of random binary (non-spin!!) starting states
msvec = np.array([ np.array([ rng.randint(2) for k in range(nspins) ], 
                           dtype=np.float)
                  for row in xrange(64) ])
# get the spin version
svec = 2*msvec - 1
print ("Initial state energy: ", 
       sorted([ sa.ClassicalIsingEnergy(v, isingJ) for v in svec ]))
print '\n'

# Generate list of nearest-neighbors for each spin
neighbors = tools.GenerateNeighbors(nspins, isingJ, 4)

# SA annealing schedule
tannealingsched = np.linspace(preannealingtemp,
                              annealingtemp,
                              annealingsteps)
# Normal annealing routine
t0 = time.time()
for vec in svec:
    sa.Anneal(tannealingsched, annealingmcsteps, vec, neighbors, rng)
t1 = time.time()
print ("Final SA energy: ", 
       sorted([ sa.ClassicalIsingEnergy(v, isingJ) for v in svec ]))
print "SA time (seconds): ", str(t1-t0)
print '\n'

# Now try multispin encoding
t0 = time.time()
sa.Anneal_multispin(tannealingsched, annealingmcsteps, msvec, neighbors, rng)
t1 = time.time()
print ("Final SA-multispin energies: ", 
       sorted([ sa.ClassicalIsingEnergy(tools.bits2spins(v), isingJ) for v in msvec ]))
print "SA-multispin time (seconds): ", str(t1-t0)
print '\n'

