## Description

Cleans the smiles from [v1.0.1](https://zenodo.org/record/3715478#.YFfpSB0pDUI) from Grambow, C. A., Pattanaik, L. & Green, W. H. Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry. Sci. data 7, 1–8 ([2020](https://www.nature.com/articles/s41597-020-0460-4)).  Some helpful scripts are also provided.

## How to Setup
After cloning the repository, the Python environment can be created via `conda env create -f environment.yml`. 
The file contains the specific package versions used during this project.

Some of the scripts utilize helpful functionality from the Reaction Mechanism Generator (RMG) software, such as [RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py), [RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database), and from the Automatic Rate Calculator ([ARC](https://github.com/ReactionMechanismGenerator/ARC)). 
Note that the `environment.yml` provided here is compatible with all three RMG repositories and can be used when compiling the cython code from RMG-Py.
