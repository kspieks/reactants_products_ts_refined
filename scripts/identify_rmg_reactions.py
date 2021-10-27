"""
Script for identifying which reactions correspond to RMG reaction templates.
Requires installation of
RMG-Py: https://github.com/ReactionMechanismGenerator/RMG-Py
RMG-database: https://github.com/ReactionMechanismGenerator/RMG-database
"""

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem

from rmgpy import settings
from rmgpy.data.rmg import RMGDatabase
from rmgpy.exceptions import AtomTypeError
import rmgpy.molecule.element as elements
from rmgpy.molecule.molecule import Atom, Bond, Molecule


def load_rmg_database():
    """
    Helper function to load the RMG-database.

    Returns:
        An instance of RMG-database
    """
    database_path = settings['database.directory']
    database = RMGDatabase()
    database.load(path=database_path,
                  thermo_libraries=['primaryThermoLibrary'],
                  reaction_libraries=[],
                  seed_mechanisms=[],
                  kinetics_families='default',
                  )
    return database


def from_rdkit_mol(rdkit_mol, sort=False, raise_atomtype_exception=True):
    """
    Converts an RDKit Molecule object to an RMG-Py Molecule object. 

    Args:
        rdkit_mol: An RDKit Molecule object
        sort: boolean indicating whether to sort the atoms in the new RMG-Py Molecule object.
              atoms are sorted by placing heaviest atoms first and H atoms last
        raise_atomtype_exception
    """
    bond_order_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'QUADRUPLE': 4, 'AROMATIC': 1.5}

    mol = Molecule()
    mol.vertices = []

    rdkit_mol.UpdatePropertyCache(strict=False)
    Chem.rdmolops.Kekulize(rdkit_mol, clearAromaticFlags=True)

    # iterate through atoms
    for i in range(rdkit_mol.GetNumAtoms()):
        rdkit_atom = rdkit_mol.GetAtomWithIdx(i)

        # use atomic number as key for element
        number = rdkit_atom.GetAtomicNum()
        isotope = rdkit_atom.GetIsotope()
        element = elements.get_element(number, isotope or -1)

        # process charge
        charge = rdkit_atom.GetFormalCharge()
        radical_electrons = rdkit_atom.GetNumRadicalElectrons()
        atom = Atom(element, radical_electrons, charge, '', 0)
        mol.vertices.append(atom)

        # add bonds by iterating again through atoms
        for j in range(0, i):
            rdkit_bond = rdkit_mol.GetBondBetweenAtoms(i, j)
            if rdkit_bond is not None:
                # process bond type
                rd_bond_type = rdkit_bond.GetBondType()
                order = bond_order_dict[rd_bond_type.name]
                bond = Bond(mol.vertices[i], mol.vertices[j], order)
                mol.add_bond(bond)

    # update lone pairs first because the charge was set by RDKit
    mol.update_lone_pairs()

    # update the atom type, charge, and multiplicity
    mol.update(raise_atomtype_exception=raise_atomtype_exception, sort_atoms=sort)

    return mol


def find_reaction_family(database, reactants, products, verbose=True):
    """
    Helper function for finding RMG reaction families when given a set of reactants and products.

    Args:
        database: an instance of RMG database.
        reactants: list of reactant molecules as RMG Molecule objects
        products: list of product molecules as RMG Molecule objects
        verbose: boolean indicating whether to print results

    Returns:
        (family_label, is_forward). None if no match.
    """
    
    # see if RMG can find this reaction
    for family in database.kinetics.families.values():
        family.save_order = False
    reaction_list = database.kinetics.generate_reactions(reactants=[mol.copy() for mol in reactants],
                                                         products=[mol.copy() for mol in products])
    # get reaction information
    for rxn in reaction_list:
        family, forward = rxn.family, rxn.is_forward
        if verbose:
            print(f'{rxn}\n',
                  f'RMG family: {family}\n',
                  f'Is forward reaction: {forward}')
        return family, forward
    else:
        if verbose:
            print("Doesn't match any RMG reaction family!")


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script to identify which reactions correspond to RMG reaction templates')
    parser.add_argument('--csv', type=str, nargs=1, default='wb97xd3_forward.csv',
                        help='path to a csv, containing idx, rsmi, psmi, ea, dh for example')

    args = parser.parse_args(command_line_args)

    return args


def main():
    # parse arguments
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # load RMG-database
    database = load_rmg_database()

    # read csv from previous step and create new column for storing RMG reaction family
    df = pd.read_csv(args.csv)    
    a = np.empty(df.shape[0])
    a[:] = np.nan
    df.insert(df.shape[1], "rmg_family", a) 

    for i in range(df.shape[0]):
        try:
            # reactant
            rsmile = df.rsmi.values[i]
            r_mol = Chem.MolFromSmiles(rsmile, sanitize=False)
            reactant_mols = [from_rdkit_mol(r_mol)]

            # product/s
            psmiles = df.psmi.values[i]
            p_mols = [Chem.MolFromSmiles(psmi, sanitize=False) for psmi in psmiles.split('.')]
            product_mols = [from_rdkit_mol(p_mol) for p_mol in p_mols] 
            
            # find the reaction in the RMG-database if it matches any existing templates
            family_label = None

            family_label, forward = find_reaction_family(database,
                                                         reactant_mols,
                                                         product_mols,
                                                         verbose=False)
        except AtomTypeError as e:
            pass
        except TypeError:
            # cannot find any matches
            pass      

        if family_label:
            df.iloc[i, -1] = family_label

    num_RMG_reactions = len(df[~df.rmg_family.isna()])
    print(f'Number of RMG reactions: {num_RMG_reactions}')

    # write results to a new file
    csv_name = f"{args.csv.split('.')[0]}_RMG_families.csv"
    df.to_csv(csv_name, index=False)


if __name__ == '__main__':
    main()
