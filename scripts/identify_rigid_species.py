"""
Uses RDKit to identify molecules that have no rotatable bonds. 
If rings are present, the rings must be planar (either aromatic or 3-membered).

https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html?highlight=maccs#rdkit.Chem.rdMolDescriptors.CalcNumRotatableBonds
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds


def isRingAromatic(mol, bond_ring):
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


def identify_rigid(args):
    """
    Function for identifying reactions whose reactant and product/s are both rigid

    Args:
        args: command line arguments
    """
    df = pd.read_csv(args.csv)
    print(f'\nTotal reactions: {df.shape[0]}')

    rigid_reactions = [None] * df.shape[0]
    for i, (rsmi, psmi) in enumerate(zip(df.rsmi.values, df.psmi.values)):
        rmol = Chem.AddHs(Chem.MolFromSmiles(rsmi))
        pmols = [Chem.AddHs(Chem.MolFromSmiles(p)) for p in psmi.split('.')]
        
        r_num_rotatable_bonds = CalcNumRotatableBonds(rmol)
        p_num_rotatable_bonds = [CalcNumRotatableBonds(pmol) for pmol in pmols]

        # if no rotatable bonds, ensure that any rings are either aromatic or 3-membered
        if r_num_rotatable_bonds + np.sum(p_num_rotatable_bonds) == 0:
            r_ring_bonds = rmol.GetRingInfo().BondRings()
            if len(r_ring_bonds) == 0:
                r_check = True
            elif len(r_ring_bonds) >= 2:
                r_check = False
            elif len(r_ring_bonds[0]) <= 3:
                r_check = True
            else:
                r_check = isRingAromatic(rmol, r_ring_bonds[0])
            
            p_check = True
            for pmol in pmols:
                p_ring_bonds = pmol.GetRingInfo().BondRings()
                if len(p_ring_bonds) == 0:
                    p_check = p_check * True
                elif len(p_ring_bonds) >= 2:
                    p_check = p_check * False
                elif len(p_ring_bonds[0]) <= 3:
                    p_check = p_check * True
                else:
                    p_check = p_check * isRingAromatic(pmol, p_ring_bonds[0])
            if r_check * p_check:
                rigid_reactions[i] = 'True'
    num_rigid = sum([x is not None for x in rigid_reactions])
    print(f'Reactions where the stable species have no rotatable bonds: {num_rigid} i.e. {num_rigid/df.shape[0]*100:.1f}%')

    df.insert(df.shape[1], 'rigid', rigid_reactions, True)
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
    parser.add_argument('--csv', type=str, nargs=1, default='wb97xd3.csv',
                        help='path to the csv containing info parsed from Arkane outputs')
    parser.add_argument('--lot', type=str, nargs=1, default='wb97xd3', 
                        help='level of theory used during the quantum calculations, used to name the new csv file')
    args = parser.parse_args(command_line_args)

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    calculate_torsion(args)


if __name__ == '__main__':
    main()
