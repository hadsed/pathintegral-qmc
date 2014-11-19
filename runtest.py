'''

File:
Author: Hadayat Seddiqi
Date: 10.19.14
Description: Test the PIQA code by collecting some data.

'''

import numpy as np
import piqa


# Define some parameters
nrows = 32
fieldstart = 1.5
fieldend = 1e-8
preannealing = True
preannealingsteps = 1
preannealingtemp = 3.0
seed = None
PTlist = [1.,1.,1.,1.,1.5,2.]
Plist = [5,10,20,40,30,40]
taulist = [5e2,5e3,5e4,5e5,5e6]

for pt in PTlist:
    for trotterslices in Plist:
        for mcsteps in taulist:
            annealingtemp = 1./trotterslices
            minenergy, minconfig = piqa.SimulateQuantumAnnealing(
                trotterslices, nrows, annealingtemp, mcsteps, fieldstart
                fieldend, preannealing, preannealinsteps, preannealingtemp,
                seed, inputfname, outputfname, False
                )
