"""
Script for separating product complexes into individual QChem input files.
Requires downloading b97d3.tar.gz and wb97xd3.tar.gz from https://zenodo.org/record/3715478#.YJgCNh0pDUJ
"""

import argparse
import os
import pandas as pd
from rdkit import Chem

from arc.parser import parse_xyz_from_file
from arc.species.converter import xyz_file_format_to_xyz, xyz_to_str, xyz_to_xyz_file_format


wb97xd3_input_script = """$molecule
   {charge} {spin}
   {xyz}
$end

$rem
   JOBTYPE                   OPT
   METHOD                    {method}
   BASIS                     {basis}
   UNRESTRICTED              TRUE
   SCF_ALGORITHM             DIIS
   MAX_SCF_CYCLES            100
   SCF_CONVERGENCE           8
   SYM_IGNORE                TRUE
   SYMMETRY                  FALSE
   GEOM_OPT_MAX_CYCLES       100
   GEOM_OPT_TOL_GRADIENT     100
   GEOM_OPT_TOL_DISPLACEMENT 400
   GEOM_OPT_TOL_ENERGY       33
   WAVEFUNCTION_ANALYSIS     FALSE
$end

@@@ 

$molecule
   read
$end

$rem
   JOBTYPE                   FREQ
   METHOD                    {method}
   BASIS                     {basis}
   UNRESTRICTED              TRUE
   SCF_GUESS                 READ
   SCF_ALGORITHM             DIIS
   MAX_SCF_CYCLES            100
   SCF_CONVERGENCE           8
   SYM_IGNORE                TRUE
   SYMMETRY                  FALSE
   WAVEFUNCTION_ANALYSIS     FALSE
$end

"""

b97d3_input_script = """$molecule
   {charge} {spin}
   {xyz}
$end

$rem
   JOBTYPE                   OPT
   METHOD                    {method}
   DFT_D                     D3_BJ
   BASIS                     {basis}
   UNRESTRICTED              TRUE
   SCF_ALGORITHM             DIIS
   MAX_SCF_CYCLES            150
   SCF_CONVERGENCE           8
   SYM_IGNORE                TRUE
   SYMMETRY                  FALSE
   GEOM_OPT_MAX_CYCLES       150
   GEOM_OPT_TOL_GRADIENT     100
   GEOM_OPT_TOL_DISPLACEMENT 400
   GEOM_OPT_TOL_ENERGY       33
   WAVEFUNCTION_ANALYSIS     FALSE
$end

@@@ 

$molecule
   read
$end

$rem
   JOBTYPE                   FREQ
   METHOD                    {method}
   DFT_D                     D3_BJ
   BASIS                     {basis}
   UNRESTRICTED              TRUE
   SCF_GUESS                 READ
   SCF_ALGORITHM             DIIS
   MAX_SCF_CYCLES            150
   SCF_CONVERGENCE           8
   SYM_IGNORE                TRUE
   SYMMETRY                  FALSE
   WAVEFUNCTION_ANALYSIS     FALSE
$end

"""
input_script = {'B97-D3': b97d3_input_script,
                'wB97X-D3': wb97xd3_input_script,
                }


def make_qchem_inputs(grambow_logs, method, basis, num_products, df_cleaned_smiles):
    """
    Parses the original Q-Chem log files from Grambow et al., identifying reactions
    with the specified number of products. Since these were previously run as one
    product complex, this function splits up the geometry into individual input files
    that can be re-run with Q-Chem at the identical settings.

    Args:
        grambow_logs: path to the original Q-Chem log files from Grambow et al.
        method: string indicating which method to use in Q-Chem
        basis: string indicating which basis to use in Q-Chem
        num_products: integer specifying how many individual products are present 
                      in the product complex that is being split up into individual input files
        df_cleaned_smiles: dataframe containing the cleaned SMILES for the reactions are the corresponding level of theory
    """

    output_dir = f'{method}_1_to_{num_products}_inputs'

    for root, dirs, files in os.walk(grambow_logs):
        dirs.sort()
        for i, directory in enumerate(dirs):
            index = directory[3:]
            r_smiles_list = df_cleaned_smiles.rsmi.values[int(index)].split('.')
            p_smiles_list = df_cleaned_smiles.psmi.values[int(index)].split('.')
              
            if len(r_smiles_list) == 1 and len(p_smiles_list) == num_products:
                log_file = os.path.join(root, directory, 'p' + index + '.log')
                product_complex_xyz = xyz_to_xyz_file_format(parse_xyz_from_file(log_file))
                xyz_line_list = []
                for line in product_complex_xyz.split('\n'):
                    xyz_line_list.append(line.strip())
          
                product_count = 0
                total_spin = 0
                for smiles in p_smiles_list:
                    atom_num = smiles.split(':')
                    atom_index_list = []
                    for i in range(1, len(atom_num)):
                        more_split = atom_num[i].split(']')
                        atom_index_list.append(int(more_split[0]))
                    separate_xyz_str = f'{len(atom_index_list)}\n{smiles}\n'
                      
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                      
                    charge = Chem.GetFormalCharge(mol)
                    if charge != 0:
                        print(f'{directory} product {product_count} had charge {charge}')
                    
                    spin = Chem.Descriptors.NumRadicalElectrons(mol)  
                    # since the reactant and TS were calculated in the singlet state, each product must be in the singlet state
                    # alternatively, both products can be spin 1 (opposite spins that give a net spin of 0)
                    if spin == 2:
                        spin = 0
                    total_spin += spin
                    if spin != 0:
                        print(f'{directory} product {product_count} had spin {spin}')
                          
                    for atom_index in atom_index_list:
                        # actual line number in xyz = atom_index + 1 (atom index starts from 1 and there are 2 additional lines in actual xyz file)
                        separate_xyz_str += xyz_line_list[atom_index+1] + '\n'
                      
                    xyz = xyz_to_str(xyz_file_format_to_xyz(separate_xyz_str))
                      
                    new_folder = os.path.join(output_dir, f'{directory}_p{product_count}')
                    os.makedirs(new_folder, exist_ok=True)
                    with open(os.path.join(new_folder, 'input.in'), 'w') as f:
                        f.write(input_script[method].format(charge=charge,
                                                            spin=spin + 1,
                                                            xyz='\n   '.join(xyz.split('\n')),
                                                            method=method,
                                                            basis=basis,
                                                            )
                                )
                      
                    product_count += 1
                      
                    if product_count == num_products:
                        if total_spin != 0 and total_spin != 2:
                            print(f'ERROR: {directory} did not have a net spin of 0 or 2!')
                        total_spin = 0


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for separating product complexes')
    parser.add_argument('--grambow_b97d3', type=str, nargs=1, default='b97d3',
                        help='path to directory containing B97-D3 Q-Chem log files from the 2020 work published by Grambow et al.')
    parser.add_argument('--b97d3_cleaned', type=str, nargs=1, default='b97d3_cleaned.csv',
                        help='csv file of cleaned smiles for B97-D3 dataset')
    
    parser.add_argument('--grambow_wb97xd3', type=str, nargs=1, default='wb97xd3',
                        help='path to directory containing wB97X-D3 Q-Chem log files from the 2020 work published by Grambow et al.')
    parser.add_argument('--wb97xd3_cleaned', type=str, nargs=1, default='wb97xd3_cleaned.csv',
                        help='csv file of cleaned smiles for wB97X-D3 dataset')

    args = parser.parse_args(command_line_args)

    return args


def main():
    """Generate input files for individual products that were previously in a product complex"""

    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # count number of b97d3 reaction directories from the original publication
    num_rxns = len(os.listdir(args.grambow_b97d3))
    print(f'\nNumber of reactions in Grambow et al. v1.0.1 B97-D3 dataset: {num_rxns}')
    df_cleaned_smiles = pd.read_csv(args.b97d3_cleaned)
    print(f'Number of reactions in the cleaned csv: {df_cleaned_smiles.shape[0]}')

    method = 'B97-D3'
    basis = 'def2-mSVP'
    for num_products in range(2, 4):
        print(f'Separating product complexes for reactions with {num_products} products...')
        make_qchem_inputs(args.grambow_b97d3, method, basis, num_products, df_cleaned_smiles)

    # count number of wb97xd3 reaction directories from the original publication
    num_rxns = len(os.listdir(args.grambow_wb97xd3))
    print(f'\nNumber of reactions in Grambow et al. v1.0.1 wB97X-D3 dataset: {num_rxns}')
    df_cleaned_smiles = pd.read_csv(args.wb97xd3_cleaned)
    print(f'Number of reactions in the cleaned csv: {df_cleaned_smiles.shape[0]}')

    method = 'wB97X-D3'
    basis = 'def2-TZVP'
    for num_products in range(2, 4):
        print(f'Separating product complexes for reactions with {num_products} products...')
        make_qchem_inputs(args.grambow_wb97xd3, method, basis, num_products, df_cleaned_smiles)


if __name__ == '__main__':
    main()
