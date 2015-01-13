# pathintegral-qmc
A path-integral quantum Monte Carlo code for simulating quantum annealing with arbitrary Ising Hamiltonians. It is written based on the 2002 Phys. Rev. B paper by Martonak, Santoro, and Tosatti entitled, 'Quantum annealing by the path-integral Monte Carlo method: The two-dimensional random Ising model' (you may find a free copy of this on arxiv.org).

## Requirements
This simulation package is written in Cython and requires scipy and numpy. Right now there isn't a good setup available and I just use pyximport for development.

## Quickstart
The fastest way to get started with this code is to run ```python piqa.py```, which will run a quick problem with some default options. To tweak the options, type ```python piqa.py --help``` to see what you can do.

## Usage
A better way to use the software is to create your own script and call the relevant functions after importing. There are a few examples given, but ```testspinglass32.py``` is probably the clearest, although the fastest one will be ```testboixo.py``` since it's only 8 qubits (it also shows how you can nicely print out the state configurations). The basic required elements of your script are as follows:

- initialize all the relevant parameters
- specify the Ising matrix (you will want to do this in sparse DOK format for maximum efficiency). you can do this by:
  - reading an input file
  - generating a random square 2D model using ```gen_ising.py```
- create the neighbors data structure, which allows efficient evaluation of the Ising energy (it requires casting the Ising matrix to DOK form but will return in DIA form)
- initialize the state
  - start with a random spin vector for a single Ising model
  - (optional) preanneal this configuration to something sensible
  - copy the spin vector to the columns of a configuration matrix
  - this will be the input to the quantum annealing routine
- run the annealing procedure

All the functions have descriptions of inputs and outputs in their docstrings (if they don't, let me know), so take a gander if you're curious.

## Contact
This is only a rudimentary guide, so undoubtedly I've left out some important things. If that is the case, please do not hesitate to contact me at [my github username] @ [gmail].