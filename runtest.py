'''

File:
Author: Hadayat Seddiqi
Date: 10.19.14
Description: Test the PIQA code by collecting some data.

'''

import pickle
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
TPlist = [ (pt/p, p) for pt,p in zip(PTlist, Plist) ]
taulist = [5e2,5e3,5e4,5e5,5e6]

dir_pref = 'ising_instances/'
inp_pref = 'inst_'
inp_suf = '.npz'
inputlist = [ dir_pref+inp_pref+str(k)+inp_suf for k in range(10) ]

# Main simulation loop
for inputnum, inputfname in enumerate(inputlist):
    TPdata = []
    for annealingtemp, trotterslices in TPlist:
        taudata = []
        for isteps, annealingsteps in enumerate(taulist):
            minenergy, minconfig = piqa.SimulateQuantumAnnealing(
                trotterslices, nrows, annealingtemp, annealingsteps, 
                fieldstart, fieldend, preannealing, preannealingsteps, 
                preannealingtemp, seed, inputfname, False
            )
            taudata.append([ annealingsteps, minenergy ])
            print("Tau", annealingsteps)
        TPdata.append([ annealingtemp, trotterslices, taudata ])
        print("T", annealingtemp, "P", trotterslices)
    # Save output file for this instance
    outdata = [ TPdata, 
                {'nrows': nrows,
                 'fieldstart': fieldstart,
                 'fieldend': fieldend,
                 'preannealing': preannealing,
                 'preannealingsteps': preannealingsteps,
                 'preannealingtemp': preannealingtemp,
                 'seed': seed} ]
    with open(inputfname[:-4]+'.pk','w') as pkfile:
        pickle.dump(outdata, pkfile)
    print("Saving instance "+str(inputnum)+".")
