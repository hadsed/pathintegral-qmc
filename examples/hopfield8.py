'''

File: hopfield8.py
Author: Hadayat Seddiqi
Date: 03.26.15
Description: Run an 8-qubit Hopfield net.

'''

import os
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse as sps
import matplotlib.pyplot as plt

import piqmc.sa as sa
import piqmc.qmc as qmc
import piqmc.tools as tools

# Define some parameters
nspins = 8
tannealingsteps = 5
tannealingmcsteps = 100
tempstart = 8.0
tempend = 1e-8
annealingtemp = 0.01
annealingsteps = 5
annealingmcsteps = 100
trotterslices = 10
fieldstart = 8.0
fieldend = 1e-8
samples = 100
# Random number generator
seed = None
rng = np.random.RandomState(seed)
# save plot or show it
saveplot = False
# network parameters
memories = [[-1,-1,-1,-1, 1, 1, 1, 1],
            [-1, 1,-1, 1,-1, 1,-1, 1],
            [-1,-1, 1, 1,-1,-1, 1, 1]]
inpbias = 0.024
# vinput = [0,0,0,0,0,0,0,0]
vinput = [-1,-1,-1,1,1,1,1,1]
# Generate annealing schedules
tannealingsched = np.linspace(tempstart,
                              tempend,
                              tannealingsteps)
annealingsched = np.linspace(fieldstart,
                             fieldend,
                             annealingsteps)
# exponential schedule
rexp = (fieldend/fieldstart)**(1./annealingsteps)
expsched = np.array([ fieldstart*rexp**k
                      for k in xrange(annealingsteps) ])
trexp = (tempend/tempstart)**(1./tannealingsteps)
texpsched = np.array([ tempstart*trexp**k
                       for k in xrange(tannealingsteps) ])
# logarithmic schedule
rlog = (fieldend/(fieldstart-1))/np.log(annealingsteps+1)
logsched = np.array([ fieldstart/(np.log(k+1)*rlog + 1)
                      for k in xrange(annealingsteps) ])
trlog = (tempend/(tempstart-1))/np.log(tannealingsteps+1)
tlogsched = np.array([ tempstart/(np.log(k+1)*trlog + 1)
                      for k in xrange(tannealingsteps) ])
# hack
tannealingsched = texpsched
annealingsched = expsched

def getbitstr(vec):
    """ Return bitstring from spin vector array. """
    return reduce(lambda x,y: x+y, 
                  [ str(int(k)) for k in tools.spins2bits(vec) ])
# Construct Ising matrix
memMat = sp.matrix(memories).T
isingJ = sp.triu(memMat * sp.linalg.pinv(memMat))
isingJ -= sp.diag(sp.diag(isingJ))
isingJ = sp.triu(memMat * sp.linalg.pinv(memMat))
isingJ += inpbias*sp.diag(vinput)
isingJ = sps.dok_matrix(isingJ)
# get energies of all states
results = []
energies = np.zeros(2**nspins)
for b in [ bin(x)[2:].rjust(nspins, '0') for x in range(2**nspins) ]:
    svec = tools.bits2spins(np.array([ int(k) for k in b ]))
    energies[int(b,2)] = sa.ClassicalIsingEnergy(svec, isingJ)
    results.append([energies[int(b,2)], b])
print("All possible states and their energies:")
for res in sorted(results)[:10]:
    print("%s    %2.4f" % tuple(res[::-1]))

# Initialize random state
spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                      dtype=np.float)
confs = np.tile(spinVector, (trotterslices, 1)).T
# Generate list of nearest-neighbors for each spin
neighbors = tools.GenerateNeighbors(nspins, isingJ, 8)
# keep a count of the populations
coinc_sa = np.zeros(2**nspins)
coinc_qa = np.zeros(2**nspins)

print("")
print("# samples", samples)

# Try using SA (random start)
for sa_itr in xrange(samples):
    spinVector = np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                          dtype=np.float)
    starten, startstate = (sa.ClassicalIsingEnergy(spinVector, isingJ), 
                           getbitstr(spinVector))
    sa.Anneal(tannealingsched, tannealingmcsteps, 
              spinVector, neighbors, rng)
    bitstr = getbitstr(spinVector)
    coinc_sa[int(bitstr, 2)] += 1

# Now do PIQA
qmc_errors_diff = 0
for s in xrange(samples):
    confs = np.tile(np.array([ 2*rng.randint(2)-1 for k in range(nspins) ], 
                             dtype=np.float), (trotterslices, 1)).T
    qmc.QuantumAnneal(annealingsched, annealingmcsteps, 
                      trotterslices, annealingtemp, nspins, 
                      confs, neighbors, rng)
    if not np.all(np.sum(confs, axis=1)/trotterslices
                  == confs[:,0]):
        qmc_errors_diff += 1
    else:
        bitstr = reduce(
            lambda x,y: x+y, 
            [ str(int(k)) for k in tools.spins2bits(confs[:,0]) ]
        )
        coinc_qa[int(bitstr, 2)] += 1

print("QMC differing slices:", qmc_errors_diff)
print("Input:", getbitstr(vinput))
print("Memories:")
for k in memories:
    print(k,getbitstr(k))

print("Coincidences (state, QA count, SA count):")
stuff = sorted(enumerate(zip(coinc_qa, coinc_sa)),
               key=lambda x: x[1][0])[::-1]
for idx, (qa, sa) in stuff[:16]:
    print("%s\t%d\t%d" % (bin(idx)[2:].rjust(8,'0'), qa, sa))

def fsaveplot(path, ext='png', close=True, verbose=True):
    """
    Save a figure from pyplot.

    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")

def plotcoinc_all(dsa, dqa, energies, ann, mcsteps, save=True):
    """ 
    Plot a bar graph of the coincidence data for all states.
    """
    dsa_total = np.sum(dsa)
    dqa_total = np.sum(dqa)

    ind = np.arange(2**nspins)
    width = 0.1
    xlim = [-20,2**nspins+20]
    maxheight = max(np.amax(dsa/dsa_total), np.amax(dqa/dqa_total))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20,15))
    fig.suptitle('Coincidences of measured states (for '+
                 str(ann)+' annealing steps and '+
                 str(mcsteps)+' Monte Carlo steps)',
                 fontsize=16, fontweight='bold')
    # common ylabel
    fig.text(0.08, 0.5, 'Coincidences', fontsize=14, fontweight='bold',
             ha='center', va='center', rotation='vertical')

    # mark the global minima in all plots
    epoints = []
    emin = np.min(energies)
    for e in sorted(enumerate(energies), key=lambda x: x[1]):
        if e[1] == emin:
            epoints.append(e[0])
    [ (ax1.axvline(ek+width, color='r', linestyle=':', linewidth=1),
       ax2.axvline(ek+width, color='r', linestyle=':', linewidth=1),
       ax3.axvline(ek+width, color='r', linestyle=':', linewidth=1))
      for ek in epoints ]
    # ax3.plot(epoints, energies[epoints], 'ro', markersize=8)
    # SA counts
    rects1 = ax1.bar(ind+width, dsa/dsa_total, width, color='r')
    ax1.set_ylim([0,1.05*maxheight])
    ax1.set_xlim(xlim)
    # ax1.set_ylabel('Coincidences', fontsize=14, fontweight='bold')
    ax1.set_title('Simulated Annealing', fontsize=12, fontweight='bold')
    # QA counts
    rects2 = ax2.bar(ind+width, dqa/dqa_total, width, color='b')
    ax2.set_ylim([0,1.05*maxheight])
    ax2.set_xlim(xlim)
    # ax2.set_ylabel('Coincidences', fontsize=14, fontweight='bold')
    ax2.set_title('Quantum Annealing', fontsize=12, fontweight='bold')
    # energies
    ax3.plot(ind+width, energies, linewidth=0.8)
    ax3.set_title('Energy Levels', fontsize=12, fontweight='bold')
    ax3.set_xlabel('States (binary converted to decimal)',
                   fontsize=14, fontweight='bold')

    if save:
        fsaveplot('figs/hopfield_'+str(ann)+'ann_'+str(mcsteps)+'mcs_coinc')
    else:
        plt.show()

plotcoinc_all(coinc_sa, coinc_qa, energies, annealingsteps, 
              annealingmcsteps, save=saveplot)
