"""
Script for creating MOLPRO input files for running high level single point calculations.
Requires installation of ARC: https://github.com/ReactionMechanismGenerator/ARC
Assumes the wb97xd3 log files have been organized as follows for unimolecular reactions:
rxn######
    p######.log
    r######.log
    ts######.log
For reactions with 2 products, the script assumes:
rxn######
    p######_0.log
    p######_1.log
    r######.log
    ts######.log
For reactions with 3 products, the script assumes:
rxn######
    p######_0.log
    p######_1.log
    p######_2.log
    r######.log
    ts######.log
"""

import argparse
import numpy as np
import os
import pandas as pd
from rdkit import Chem

from arc.parser import parse_xyz_from_file
from arc.species.converter import xyz_to_str


ccsdf12_input_script = """***,name
memory,1152,m;
geometry={{angstrom;
{xyz}}}

basis={basis}

int;

{{hf;
maxit,300;
wf,spin={spin},charge={charge};}}

{method};


---;


"""


def write_molpro_input(args, log_file, output_directory, charge=0, spin=0):
    """
    Writes a molpro input file.

    Args:
        args: command line arguments
        log_file: string specifying the log file path of a single species
        output_directory: string specifying the output directory to write the new input file
        charge: integer denoting the species charge
        spin: integer denoting the species spin
    """
    # parse the optimized geometry
    xyz = parse_xyz_from_file(log_file)

    # create molpro input file
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, 'input.in'), 'w') as f:
        f.write(ccsdf12_input_script.format(charge=charge,
                                            spin=spin,
                                            xyz=xyz_to_str(xyz),
                                            method=args.method,
                                            basis=args.basis,
                                            )
                )


def make_inputs(args, num_products, df_cleaned_smiles, species):
    """
    Creates MOLPRO input files for TS and products for running high level single point calculations.

    Args:
        args: command line arguments
        num_products: integer specifying how many individual products are present 
                      in the product complex that is being split up into individual input files
        df_cleaned_smiles: dataframe containing the cleaned SMILES for the reactions are the corresponding level of theory
        species: One of 'reactant', 'ts', or 'product' to indicate which geometry to parse for the input file
    """
    # many reactions start from the same reactant so only run the unique ones to save computational effort
    unique_reactants = df_cleaned_smiles.rsmi.unique()

    for root, dirs, files in os.walk(args.wb97xd3_logs):
        dirs.sort()
        for directory in dirs:
            index = directory[3:]        
            r_smiles_list = df_cleaned_smiles.rsmi.values[int(index)].split('.')
            p_smiles_list = df_cleaned_smiles.psmi.values[int(index)].split('.')

            write = False
            if (len(r_smiles_list) == 1) and (len(p_smiles_list) == num_products):
                # parse the optimized geometry
                if (species == 'reactant') and (r_smiles_list[0] in unique_reactants):
                    # remove the reactant so the input file is only created once
                    unique_reactants = np.delete(unique_reactants, np.where(unique_reactants == r_smiles_list[0]))
                    charge, spin, write = 0, 0, True
                    log_file = os.path.join(root, directory, 'r' + index + '.log')
                    output_dir = f'{args.method}_1_to_{num_products}_unique_reactant_inputs'
                
                elif species == 'ts':
                    charge, spin, write = 0, 0, True
                    log_file = os.path.join(root, directory, 'ts' + index + '.log')
                    output_dir = f'{args.method}_1_to_{num_products}_ts_inputs'
                
                elif (species == 'product') and (num_products == 1):
                    charge, spin, write = 0, 0, True
                    log_file = os.path.join(root, directory, 'p' + index + '.log')
                    output_dir = f'{args.method}_1_to_{num_products}_product_inputs'
                
                elif (species == 'product') and (num_products > 1):
                    # use RDKit to determine the charge and spin for the multiple products
                    product_count = 0
                    total_spin = 0
                    for smiles in p_smiles_list:
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

                        # write molpro input file
                        log_file = os.path.join(root, directory, f'p{index}_{product_count}' +  '.log')
                        output_dir = f'{args.method}_1_to_{num_products}_product_inputs'
                        output_directory = os.path.join(output_dir, directory + f'_p{product_count}')
                        write_molpro_input(args, log_file, output_directory, charge=charge, spin=spin)

                        product_count += 1
                        if product_count == num_products:
                            if total_spin != 0 and total_spin != 2:
                                print(f'ERROR: {directory} did not have a net spin of 0 or 2!')
                            total_spin = 0

                if write:
                    # write molpro input file
                    output_directory = os.path.join(output_dir, directory)
                    write_molpro_input(args, log_file, output_directory, charge=charge, spin=spin)

        break


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for creating molpro input files')
    parser.add_argument('--wb97xd3_logs', type=str, nargs=1, default='wb97xd3/qm_logs', 
                        help='folder containing log files for wB97X-D3 dataset')
    parser.add_argument('--method', type=str, nargs=1, default='ccsd(t)-f12',
                        help='method to be used with MOLPRO calculation')
    parser.add_argument('--basis', type=str, nargs=1, default='cc-pvdz-f12', 
                        help='basis to be used with MOLPRO calculation')
    parser.add_argument('--wb97xd3_cleaned', type=str,  nargs=1, default='wb97xd3_cleaned.csv',
                        help='csv file with cleaned smiles for wB97X-D3 dataset')
    
    args = parser.parse_args(command_line_args)

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    df_cleaned_smiles = pd.read_csv(args.wb97xd3_cleaned)
    print(f'\nNumber of reactions in the cleaned csv: {df_cleaned_smiles.shape[0]}')

    # count number of wb97xd3 reaction directories 
    num_rxns = len(os.listdir(args.wb97xd3_logs))
    print(f'\nNumber of reactions from re-optimized wB97X-D3 dataset: {num_rxns}')

    for num_products in range(1, 4):
        print(f'\nGenerating input files for reactants for reactions with {num_products} products...')
        make_inputs(args, num_products, df_cleaned_smiles, species='reactant')

        print(f'\nGenerating input files for ts for reactions with {num_products} products...')
        make_inputs(args, num_products, df_cleaned_smiles, species='ts')

        print(f'\nGenerating input files for products for reactions with {num_products} products...')
        make_inputs(args, num_products, df_cleaned_smiles, species='product')


if __name__ == '__main__':
    main()
