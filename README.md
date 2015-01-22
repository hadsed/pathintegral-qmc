# pathintegral-qmc
A path-integral quantum Monte Carlo code for simulating quantum annealing with arbitrary Ising Hamiltonians. It is written based on the 2002 Phys. Rev. B paper by Martonak, Santoro, and Tosatti entitled, 'Quantum annealing by the path-integral Monte Carlo method: The two-dimensional random Ising model' (you may find a free copy of this on arxiv.org).

## Requirements
This simulation package is written in Cython and requires ```scipy``` and ```numpy```. The C files are included with the .pyx. Installation requires ```setuptools```. There is an example that requires ```mpi4py```, but this is not a general requirement.

## Installation
After cloning the repo, navigate to where you see ```setup.py``` and run ```python setup.py install```, or if you're developing (or wish to uninstall later) do ```python setup.py develop``` (where you can write ```python setup.py develop --uninstall``` if you wish to remove it later).

## Testing
After installation, simply type ```nosetests --verbose``` to see the tests run (and hopefully succeed).

## Usage
At the top of your code, you should be importing the following: ```numpy```, ```scipy.sparse```, ```piqmc.sa```, ```piqmc.qmc```, ```piqmc.tools```. There are a few examples given in the ```examples/``` directory, with ```spinglass32.py``` probably being the clearest, although the fastest one will be ```boixo.py``` since it's only 8 qubits (it also shows how you can nicely print out the state configurations). You can also look at ```profiler.py``` which is a barebones script that just looks to profile the simulated annealing and quantum annealing routines for comparison with cProfile. You can run any of these examples by navigating to the examples directory and typing ```python boixo.py```, or whatever the name is, and it should give coherent results (the data files are in ```examples/ising_instances/```). The basic required elements of your script are as follows:

- initialize all the relevant parameters
- specify the Ising matrix (you will want to do this in sparse DOK format for maximum efficiency). you can do this by:
  - reading an input file
  - generating a random square 2D model using ```tools.Generate2DIsingInstance```
- create the neighbors data structure, which allows efficient evaluation of the Ising energy (it requires casting the Ising matrix to DOK form but will cast it back to DIA form)
- initialize the state
  - start with a random spin vector for a single Ising model
  - (optional) preanneal this configuration to something sensible
  - copy the spin vector to the columns of a configuration matrix
  - this will be the input to the quantum annealing routine
- run the annealing procedure

All the functions have descriptions of inputs and outputs in their docstrings (if they don't or they are insufficient, let me know).

If you want to use MPI to run independent simulations on several worker processors, you need to have ```mpi4py``` installed. Then you can run the ```spinglass32_mpi.py``` example by writing ```mpirun -n 4 python spinglass32_mpi.py```.

There are also a few examples that test a Cython/OpenMP parallelized version of the annealing routines (where the innermost loop of spin updating work is broken across threads), but they are still in development, they don't use randomized permutations of spin updates, and currently they are rather slow as they're naive implementations of the sequential algorithm.

## Contact
This is only a rudimentary guide, so undoubtedly I've left out some important things. If that is the case, please do not hesitate to contact me at [my github username] @ [gmail].