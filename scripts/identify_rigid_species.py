"""
Uses RDKit to identify molecules that have no rotatable bonds. 
If rings are present, the rings must be planar (either aromatic or 3-membered).
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts


def is_ring_aromatic(mol, bond_ring):
    """
    Helper function for identifying if a bond is aromatic

    Args:
        mol: rdkit molecule 
        bond_ring: rdkit bond from a ring

    Returns:
        True if the bond is in an aromatic ring. False otherwise
    """
    for idx in bond_ring:
        if not mol.GetBondWithIdx(idx).GetIsAromatic():
            return False
    return True


def is_ring_rigid(mol, ring_bonds):
    """
    Helper function for identifying if ring is rigid i.e. either aromatic or 3-membered

    Args:
        mol: rdkit molecule
        ring_bonds: tuple containing the indices specifying which bonds are in a ring 
    """
    rigid = False
    if len(ring_bonds) >= 2:
        rigid = False
    elif len(ring_bonds[0]) <= 3:
        rigid = True
    else:
        rigid = is_ring_aromatic(mol, ring_bonds[0])

    return rigid

def identify_rigid(args):
    """
    Function for identifying reactions whose reactant and product/s are both rigid

    Args:
        args: command line arguments
    """
    df = pd.read_csv(args.csv)
    print(f'\nTotal reactions: {df.shape[0]}')

    params = Chem.SmilesParserParams()
    params.removeHs = False

    rigid_reactions = [None] * df.shape[0]
    methyl_rxns = [None] * df.shape[0]
    methyl_rotor = Chem.MolFromSmarts('[CH3]')

    for i, row in df.iterrows():
        rmol = Chem.MolFromSmiles(row.rsmi, params)
        pmols = [Chem.MolFromSmiles(p, params) for p in row.psmi.split('.')]
        
        r_num_rotatable_bonds = len(rmol.GetSubstructMatches(RotatableBondSmarts))
        p_num_rotatable_bonds = sum([len(pmol.GetSubstructMatches(RotatableBondSmarts)) for pmol in pmols])

        # if no rotatable bonds, ensure that any rings are either aromatic or 3-membered so there are no ring conformers
        if r_num_rotatable_bonds + p_num_rotatable_bonds == 0:
            r_rigid = False
            r_ring_bonds = rmol.GetRingInfo().BondRings()
            if len(r_ring_bonds) == 0:
                r_rigid = True      # no rings and no rotatable bonds
            else:
                r_rigid = is_ring_rigid(rmol, r_ring_bonds)
            
            p_rigid = True
            for pmol in pmols:
                p_ring_bonds = pmol.GetRingInfo().BondRings()
                if len(p_ring_bonds) == 0:
                    p_rigid = p_rigid * True    # no rings and no rotatable bonds
                else:
                    p_rigid = p_rigid * is_ring_rigid(pmol, p_ring_bonds)

            if r_rigid * p_rigid:
                rigid_reactions[i] = 'True'

        # also accept rotatable bonds if they are just methyl rotors
        elif (r_num_rotatable_bonds > 0) & (len(rmol.GetSubstructMatches(methyl_rotor)) == r_num_rotatable_bonds):
            r_rigid = False
            r_ring_bonds = rmol.GetRingInfo().BondRings()
            if len(r_ring_bonds) == 0:
                r_rigid = True      # no rings and no rotatable bonds
            else:
                r_rigid = is_ring_rigid(rmol, r_ring_bonds)

            if r_rigid and (p_num_rotatable_bonds > 0):
                # if each rotor in the product/s are also only methyl rotors, then accept the conformers for this rxn
                if sum([len(pmol.GetSubstructMatches(methyl_rotor)) for pmol in pmols]) == p_num_rotatable_bonds:
                    # any rings in the product/s should be planar
                    p_rigid = True
                    for pmol in pmols:
                        p_ring_bonds = pmol.GetRingInfo().BondRings()
                        if len(p_ring_bonds) == 0:
                            p_rigid = p_rigid * True    # no rings and no rotatable bonds
                        else:
                            p_rigid = p_rigid * is_ring_rigid(pmol, p_ring_bonds)

                    if r_rigid * p_rigid:
                        methyl_rxns[i] = True

    num_rigid = sum([x is not None for x in rigid_reactions])
    print(f'Reactions where the stable species have no rotatable bonds: {num_rigid} i.e. {num_rigid/df.shape[0]*100:.1f}%')
    df.insert(df.shape[1], 'rigid', rigid_reactions, True)

    num_methyl = sum([x is not None for x in methyl_rxns])
    print(f'Reactions where the stable species only have methyl rotors as rotatable bonds: {num_methyl} i.e. {num_methyl/df.shape[0]*100:.1f}%')
    df.insert(df.shape[1], 'methyl', methyl_rxns)

    df.to_csv(f'{args.lot}_cleaned_rigid_reactions.csv', index=False)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script to identify number of reactions with rigid species')
    parser.add_argument('--csv', type=str, default='wb97xd3.csv',
                        help='path to the csv containing info parsed from Arkane outputs')
    parser.add_argument('--lot', type=str, default='wb97xd3', 
                        help='level of theory used during the quantum calculations, used to name the new csv file')
    args = parser.parse_args(command_line_args)

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    identify_rigid(args)


if __name__ == '__main__':
    main()
