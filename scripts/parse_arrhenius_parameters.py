"""
Parses the fitted A-factor and activation energy from Arkane.
Compiles these values with the reactant and product SMILES 
along with the reaction index into a csv file.
"""

import argparse
import numpy as np
import os
import pandas as pd
import re
import shutil


def get_fwd_rates(args, num_rxns, df_cleaned_smiles):
    """
    Function for parsing the forward rate parameters from Arkane output files.

    Args:
        args: command line arguments
        num_rxns: integer denoting the number of reactons (directories) from Arkane
        df_cleaned_smiles: dataframe containing the cleaned smiles of the reactants and products
    """
    cols = ['idx', 'A', 'Ea']
    df = pd.DataFrame(np.zeros((num_rxns, len(cols))), columns=cols)
    failed_indices = []
    
    indices = {1: {'num_items': 6,  'A': 3, 'Ea': 5, 'n': 4,},
               # if 1 product, expect .split() to give ['r1', '<=>', 'p1', '5.268e+11', '1.005', '49.586'] for example
               2: {'num_items': 8,  'A': 5, 'Ea': 7, 'n': 6,},
               # if 2 products, expect .split() to give ['r1', '<=>', 'p1', '+', 'p2', '6.014e+13', '0.000', '61.330']
               3: {'num_items': 10, 'A': 7, 'Ea': 9, 'n': 8, },
               # if 3 products, expect .split() to give ['r1', '<=>', 'p1', '+', 'p2', '+', 'p3',  '5.268e+11', '1.005', '49.586']
               }

    # parse the fitted Arrhenius parameters
    for root, dirs, files in os.walk(args.arkane_output):
        dirs.sort()
        for i, directory in enumerate(dirs):
            index = directory[3:]
            # get number of products
            for root1, dirs1, files1 in os.walk(os.path.join(root, directory, 'Species')):
                num_products = len(dirs1) - 1
                break

            chemkin_file = os.path.join(root, directory, 'rxns', 'ts'+index, 'arkane', 'chem.inp')
            df.iloc[i, 0] = str(index)  # idx
            try:
                with open(chemkin_file, 'r') as f:
                    # skip the first line
                    f.readline()
                    line = f.readline()
                    items = line.split()
                    if len(items) != indices[num_products]['num_items']:
                        print(f"Error: {directory} does not have {indices[num_products]['num_items']} items!")
                    df.iloc[i, 1] = float(items[indices[num_products]['A']])    # A
                    df.iloc[i, 2] = float(items[indices[num_products]['Ea']])   # Ea in kcal/mol
                    
                    # verify the 2-parameter Arrhenius was used
                    n = float(items[indices[num_products]['n']])
                    if n != 0:
                        print(f'Error: {directory} did not have n=0')

            # if Arkane failed to fit Arrhenius parameters for this reaction, chem.inp will be missing
            # failures were caused for a small percentage of reactions if TS energy < energy of one of the stable species
            except FileNotFoundError as e:
                print(e)
                failed_indices.append(i)
        break

    # confirm that all values were parsed properly
    print(f'{len(failed_indices)} reactions errored while parsing the forward reactions')
    if len(failed_indices) > 0:
        dst = f'{args.lot}_arkane_logs_failed'
        os.makedirs(dst, exist_ok=True)
        for root, dirs, files in os.walk(args.arkane_output):
            dirs.sort()
            for i in failed_indices:
                directory = dirs[i]
                print(f'Removing {directory} since Arkane could not generate kinetics for this reaction...')
                path = os.path.join(root, directory)
                shutil.move(src, dst)
            break

    if df[df.Ea==0].size > 0:
        print(f'Ea was not parsed for {len(df[df.Ea==0].idx.values)} reactions...')
    if df[df.A==0].size > 0:
        print(f'A factor was not parsed for {len(df[df.A==0].idx.values)} reactions...')
    df.Ea = df.Ea.replace(0, np.nan)
    df.A = df.A.replace(0, np.nan)

    df.idx = df.idx.astype('int64')
    
    # add the cleaned smiles for these reactions
    df.insert(1, 'rsmi', df_cleaned_smiles.rsmi.values, True)
    df.insert(2, 'psmi', df_cleaned_smiles.psmi.values, True)
    
    # store ln(A) rather than the raw A-factor
    df_lnA = df.copy()
    df_lnA.A = np.log(df.A)
    df_lnA.rename(columns={'A': 'lnA'}, inplace=True)
    
    # save the results
    df_lnA.to_csv(f'{args.lot}_forward.csv', index=False)


def get_rev_rates(args, num_rxns, df_cleaned_smiles):
    """
    Function for parsing the reverse rate parameters from Arkane output files.

    Args:
        args: command line arguments
        num_rxns: integer denoting the number of reactons (directories) from Arkane
        df_cleaned_smiles: dataframe containing the cleaned smiles of the reactants and products
    """
    cols = ['idx', 'A_rev', 'Ea_rev']
    df_rev = pd.DataFrame(np.zeros((num_rxns, len(cols))), columns=cols)

    # parse the fitted Arrhenius parameters
    for root, dirs, files in os.walk(args.arkane_output):
        dirs.sort()
        for i, directory in enumerate(dirs):
            index = directory[3:]
            output_file = os.path.join(root, directory, 'rxns', 'ts'+index, 'arkane', 'output.py')
            df_rev.iloc[i, 0] = index
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    # start from the bottom
                    for line in reversed(lines):
                        if 'k_rev (TST+T)' in line:
                            items = line.split('=')

                            # nearly all reactions have a large A factor written in scientific format
                            # one reaction from B97-D3 has a smaller A factor that is not in scientific format
                            try:
                                A = float(re.search("\d\.\d*e\+\d*", line[:95])[0])
                            except TypeError as e:
                                print(e)
                                A = float(re.search("A\=\(-?\d*\.?\d*",line[:95])[0][3:])
                                print(f'Warning: A factor for the reverse reaction of {directory} was not in scientific format.\n',
                                      f'Using {A}')

                            Ea = float(re.search("Ea\=\(-?\d*\.?\d*",line[:95])[0][4:])
                            Ea = Ea / 4.184  # convert from kJ/mol to kcal/mol
                            n = float(re.search("n\=\d", line[:95])[0][2])
                            
                            # verify the 2-parameter Arrhenius was used
                            if n != 0:
                                print(f'ERROR: {directory} did not have n=0')
                                break
                            else:
                                df_rev.iloc[i, 1] = A 
                                df_rev.iloc[i, 2] = Ea
                                break
            except FileNotFoundError as e:
                print(e)
                print(i)
        break

    # confirm that all values were parsed properly
    if df_rev[df_rev.Ea_rev==0].size > 0:
        print(f'Ea was not parsed for {len(df_rev[df_rev.Ea_rev==0].idx.values)} reactions...')
    if df_rev[df_rev.A_rev==0].size > 0:
        print(f'A factor was not parsed for {len(df_rev[df_rev.A_rev==0].idx.values)} reactions...')
    df_rev.Ea_rev = df_rev.Ea_rev.replace(0, np.nan)
    df_rev.A_rev = df_rev.A_rev.replace(0, np.nan)

    df_rev.idx = df_rev.idx.astype('int64')
    
    # add the cleaned smiles for these reactions
    df_rev.insert(1, 'rsmi', df_cleaned_smiles.psmi.values, True)
    df_rev.insert(2, 'psmi', df_cleaned_smiles.rsmi.values, True)

    # store ln(A) rather than the raw A-factor
    df_lnA = df_rev.copy()
    df_lnA.A_rev = np.log(df_rev.A_rev)
    df_lnA.rename(columns={'A_rev': 'lnA_rev'}, inplace=True)

    # save the results
    df_lnA.to_csv(f'{args.lot}_reverse.csv', index=False)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for parsing Arkane output to obtain Arrhenius parameters')
    parser.add_argument('--arkane_output', type=str, nargs=1, default='ccsdtf12/arkane_logs',
                        help='path to directory containing Arkane output files')
    parser.add_argument('--lot', type=str, nargs=1, default='ccsdtf12',
                        help='level of theory used during the quantum calculations, used to name the new csv file')
    parser.add_argument('--cleaned_csv', type=str, nargs=1, default='wb97xd3_cleaned.csv',
                        help='csv file of cleaned smiles for the respective level of theory')
  
    args = parser.parse_args(command_line_args)

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # count number of reaction directories
    num_rxns = len(os.listdir(args.arkane_output))
    print(f'\nNumber of reactions: {num_rxns}')

    df_cleaned_smiles = pd.read_csv(args.cleaned_csv)
    get_rev_rates(args, num_rxns, df_cleaned_smiles)
    get_fwd_rates(args, num_rxns, df_cleaned_smiles)


if __name__ == '__main__':
    main()
