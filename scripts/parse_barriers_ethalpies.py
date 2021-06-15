"""
This script determines the barrier height and enthalpy for each reaction.
Requires installation of RMG-Py: https://github.com/ReactionMechanismGenerator/RMG-Py
"""

import argparse
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd

from arkane.ess.factory import ess_factory


def get_barriers(directory, args, df_cleaned):
    """
    Computes the barrier height for a reaction by adding the zero-point energies 
    to the reactant and TS energies and taking the difference between resulting TS and reactant energies
    
    Args:
        directory: reaction subdirectory
        args: command line arguments
        df_cleaned: dataframe containing cleaned smiles
    
    Returns:
        idx_int: reaction index as an integer
        rsmi: SMILES for the reactant
        psmi: SMILES for the product/s
        ea: barrier height for the reaction in kcal/mol
    """

    idx = directory[3:]
    idx_int = int(idx)
    rsmi = df_cleaned[df_cleaned.idx == idx_int].rsmi.values[0]
    psmi = df_cleaned[df_cleaned.idx == idx_int].psmi.values[0]

    # parse the single point energy
    r_log = os.path.join(args.sp_dir, directory, f'r{idx}.log')
    r_energy = ess_factory(r_log).load_energy()/4184    # convert J/mol to kcal/mol
    
    ts_log = os.path.join(args.sp_dir, directory, f'ts{idx}.log')
    ts_energy = ess_factory(ts_log).load_energy()/4184  # convert J/mol to kcal/mol
    
    # parse the ZPE
    r_log = os.path.join(args.opt_freq_dir, directory, f'r{idx}.log')
    r_zpe = ess_factory(r_log).load_zero_point_energy()/4184       # convert J/mol to kcal/mol

    ts_log = os.path.join(args.opt_freq_dir, directory, f'ts{idx}.log')
    ts_zpe = ess_factory(ts_log).load_zero_point_energy()/4184     # convert J/mol to kcal/mol

    ea = (ts_energy + ts_zpe) - (r_energy + r_zpe)

    return [idx_int, rsmi, psmi, ea]


def get_enthalpies(directory, args):
    """
    Computes the enthalpy for a reaction by adding the zero-point energies 
    to the reactant and product energies and taking the difference between resulting product and reactant energies
    
    Args:
        directory: reaction subdirectory
        args: command line arguments

    Returns:
        dh: enthalpy for the reaction in kcal/mol
    """    
    # get the number of products
    for root, dirs, files in os.walk(os.path.join(args.sp_dir, directory)):
        num_files = len(files)
        num_products = num_files - 2
        break

    idx = directory[3:]
    # parse the single point energy for the reactant
    r_log = os.path.join(args.sp_dir, directory, f'r{idx}.log')
    r_energy = ess_factory(r_log).load_energy()/4184    # convert J/mol to kcal/mol
    
    # parse the single point energy for the product/s
    p_energies = np.zeros(num_products)
    if num_products > 1:
        for j in range(num_products):
            p_log = os.path.join(args.sp_dir, directory, f'p{idx}_{j}.log')
            p_energies[j] = ess_factory(p_log).load_energy()/4184  # convert J/mol to kcal/mol
    else:
        p_log = os.path.join(args.sp_dir, directory, f'p{idx}.log')
        p_energies[0] = ess_factory(p_log).load_energy()/4184  # convert J/mol to kcal/mol

    # parse the ZPE for the reactant
    r_log = os.path.join(args.opt_freq_dir, directory, f'r{idx}.log')
    r_zpe = ess_factory(r_log).load_zero_point_energy()/4184       # convert J/mol to kcal/mol
    
    # parse the ZPE for the product/s
    p_zpes = np.zeros(num_products)
    if num_products > 1:
        for j in range(num_products):
            p_log = os.path.join(args.opt_freq_dir, directory, f'p{idx}_{j}.log')
            p_zpes[j] = ess_factory(p_log).load_zero_point_energy()/4184  # convert J/mol to kcal/mol
    else:
        p_log = os.path.join(args.opt_freq_dir, directory, f'p{idx}.log')
        p_zpes[0] = ess_factory(p_log).load_zero_point_energy()/4184  # convert J/mol to kcal/mol

    dh = (p_energies.sum() + p_zpes.sum()) - (r_energy + r_zpe)
    return dh


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:

        The parsed command-line arguments by key words.
    """
    parser = argparse.ArgumentParser(description='Script for running Arkane to obtain rates')
    parser.add_argument('--opt_freq_dir', type=str, nargs=1, default='wb97xd3/qm_logs',
                        help='directory containing the Q-Chem logs from geometry optimization and frequency calculation')
    parser.add_argument('--sp_dir', type=str, nargs=1, default=None,
                        help='directory containing the MOLPRO logs from refined singled point calculation')
    parser.add_argument('--cleaned_csv', type=str, nargs=1, default='wb97xd3_cleaned.csv',
                        help='csv file of cleaned smiles')
    parser.add_argument('--out_name', type=str, nargs=1, default='wb97xd3_barriers_enthalpies.csv',
                        help='filename used for the output csv file')
    parser.add_argument('--n_cpus', type=int, nargs=1, default=4,
                        help='number of cpus to use while parsing the files')
    args = parser.parse_args(command_line_args)

    # if no single point directory is specified, use the single point from the geometry optimization
    if (args.sp_dir is None):
        args.sp_dir = args.opt_freq_dir

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # count number of reaction directories
    num_rxns = len(os.listdir(args.sp_dir))
    print(f'\nNumber of reactions: {num_rxns}')

    df_cleaned = pd.read_csv(args.cleaned_csv)
    for root, dirs, files in os.walk(args.sp_dir):
        dirs.sort()
        break

    print('Computing barrier heights...')
    output = Parallel(n_jobs=args.n_cpus)(delayed(get_barriers)(directory, args, df_cleaned) for directory in dirs)
    df = pd.DataFrame(output, columns=['idx', 'rsmi', 'psmi', 'ea'])

    print('Computing enthalpies...')
    dhs = Parallel(n_jobs=args.n_cpus)(delayed(get_enthalpies)(directory, args) for directory in dirs)
    df.insert(df.shape[1], 'dh', dhs)

    df.to_csv(args.out_name, index=False)


if __name__ == '__main__':
    main()
