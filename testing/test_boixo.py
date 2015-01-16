'''

File: test_boixo.py
Author: Hadayat Seddiqi
Date: 01.16.15
Description: Test the accuracy of PIQMC code by running the
             8-qubit diamond graph problem from Boixo et al.
             See: arXiv:1212.1739 [quant-ph]

'''

import numpy as np
import scipy.sparse as sps

import piqmc.sa as sa
import piqmc.qmc as qmc
import piqmc.tools as tools


class TestBoixo:
    def setUp(self):
        # Define some parameters
        self.nspins = 8
        self.preannealingtemp = 1.0
        self.annealingtemp = 0.01
        self.annealingsteps = 10
        self.annealingmcsteps = 3
        self.trotterslices = 5
        self.fieldstart = 0.5
        self.fieldend = 1e-8
        # Random number generator
        self.rng = np.random.RandomState(123)
        # Construct Ising matrix
        kvp = [
            [1, 2, 1.0],
            [1, 4, 1.0],
            [1, 5, 1.0],
            [2, 3, 1.0],
            [2, 6, 1.0],
            [3, 4, 1.0],
            [3, 7, 1.0],
            [4, 8, 1.0],
            [1, 1, 1.0],
            [2, 2, 1.0],
            [3, 3, 1.0],
            [4, 4, 1.0],
            [5, 5, -1.0],
            [6, 6, -1.0],
            [7, 7, -1.0],
            [8, 8, -1.0]
            ]
        self.isingJ = sps.dok_matrix((self.nspins,self.nspins))
        for i,j,val in kvp:
            self.isingJ[i-1,j-1] = val
        # get the neighbors data structure (tested elsewhere)
        self.nbs = tools.GenerateNeighbors(self.nspins,
                                           self.isingJ,
                                           4, None)

    def test_classicalisingenergy(self):
        svecs = np.array([
                [-1,1,-1,1,1,-1,-1,1],   # E = 4.0
                [1,1,1,1,1,-1,1,-1],     # E = -8.0
                [1,-1,1,1,-1,-1,-1,-1],  # E = -4.0
                [1,-1,-1,1,1,-1,1,1]     # E = 0.0
                ])
        energies = [4.0, -8.0, -4.0, 0.0]
        for vec, en in zip(svecs, energies):
            print sa.ClassicalIsingEnergy(vec, self.isingJ), en
            assert(sa.ClassicalIsingEnergy(vec, self.isingJ) == en)

    def test_sa(self):
        for i in range(self.trotterslices):
            spinVector = np.array([ 2*self.rng.randint(2)-1 
                                    for k in range(self.nspins) ], 
                                  dtype=np.float64)
            tannealingsched = np.linspace(self.preannealingtemp,
                                          self.annealingtemp,
                                          self.annealingsteps)
            sa.Anneal(tannealingsched, 
                      self.annealingmcsteps, 
                      spinVector, 
                      self.nbs,
                      self.rng)
            # with the given parameters, they have surely found a ground state
            assert(np.sum(spinVector[:4]) == 4 or
                   np.sum(spinVector) == -8)
            assert(sa.ClassicalIsingEnergy(spinVector, self.isingJ) == -8.0)

    def test_qmc(self):
        configurations = np.tile(np.array([ 2*self.rng.randint(2)-1 
                                            for k in range(self.nspins) ], 
                                          dtype=np.float64), 
                                 (self.trotterslices, 1)).T
        annealingsched = np.linspace(self.fieldstart,
                                     self.fieldend,
                                     self.annealingsteps)
        qmc.QuantumAnneal(annealingsched, 
                          self.annealingmcsteps, 
                          self.trotterslices, 
                          self.annealingtemp, 
                          self.nspins, 
                          configurations, 
                          self.nbs, 
                          self.rng)
        energies = np.array(
            [ sa.ClassicalIsingEnergy(configurations[:,k], self.isingJ)
              for k in range(self.trotterslices) ]
            )
        # assert that we found at least one lowest state energy amongst the slices
        assert(np.sum(energies) == -8.0*self.trotterslices)
        for k in range(self.trotterslices):
            assert(np.sum(configurations[:4,k]) == 4.0 or
                   np.sum(configurations[:,k]) == -8.0)
