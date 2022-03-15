## Description

Cleans the smiles from [v1.0.1](https://zenodo.org/record/3715478#.YFfpSB0pDUI) from Grambow, C. A., Pattanaik, L. & Green, W. H. Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry. Sci. data 7, 1–8 ([2020](https://www.nature.com/articles/s41597-020-0460-4)).  Some helpful scripts are also provided.

## How to Setup
Some of the scripts utilize helpful functionality from the Reaction Mechanism Generator (RMG) software, such as [RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py), [RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database), and from the Automatic Rate Calculator ([ARC](https://github.com/ReactionMechanismGenerator/ARC)). Clone these repositories into the same folder. 
The installation instructions for RMG-Py and RMG-database can be found [here](http://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/anacondaDeveloper.html) and involves creating the `rmg_env` and running `make` to compile RMG-Py. To install ARC, clone the repo, add ARC to your local path, and create the `arc_env`; additional detail can be found [here](https://reactionmechanismgenerator.github.io/ARC/installation.html). All code in this repo can be run within the `arc_env`.

