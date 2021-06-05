"""
Parses the rate constants calculated from TST via Arkane.
Compiles these values with the reactant and product SMILES 
along with the reaction index into a csv file.
"""

import argparse
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
import re
import shutil


def parse_rates(args, index, df_cleaned):
    """
    Parses the rate constants for the forward and reverse reactions

    Args:
        args: command line arguments
        index: reaction index as a string
        df_cleaned: dataframe containing cleaned smiles
    """
    output_file = os.path.join(args.arkane_output, f'rxn{index}', 'rxns', 'ts'+index, 'arkane', 'output.py')
    # initialize empty array
    tst_rates = []

    idx_int = int(index)
    rsmi = df_cleaned[df_cleaned.idx == idx_int].rsmi.values[0]
    psmi = df_cleaned[df_cleaned.idx == idx_int].psmi.values[0]

    tst_rates.append(idx_int)
    tst_rates.append(rsmi)
    tst_rates.append(psmi)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # parse the forward TST rate with tunnelling correction for the 50 Temperature points 
            if 'k (TST)' in line:
                for j in range(2, 52):
                    items = lines[i+j].split()
                    # check that all lines have the same format to ensure the correct value is parsed
                    assert len(items) == 7
                    k = items[-2]
                    tst_rates.append(float(k))

            if 'k_rev (TST)' in line:
                for j in range(2, 52):
                    items = lines[i+j].split()
                    k = items[-2]
                    tst_rates.append(float(k))

                break

    return tst_rates


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for parsing calculated rates')
    parser.add_argument('--arkane_output', type=str, nargs=1,
                        default='ccsdtf12/arkane_logs',
                        help='path to directory containing Arkane output files')
    parser.add_argument('--out_name', type=str, nargs=1, 
                        default='ccsdtf12_tst_rates.csv', 
                        help='level of theory used during the quantum calculations, used to name the new csv file')
    parser.add_argument('--cleaned_csv', type=str, nargs=1, 
                        default='wb97xd3_cleaned.csv',
                        help='csv file of cleaned smiles for the respective level of theory')
    parser.add_argument('--n_cpus', type=int, default=4, nargs=1, 
                        help='number of cpus to use while parsing the files')
  
    args = parser.parse_args(command_line_args)

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # count number of reaction directories
    indices = []
    for root, dirs, files in os.walk(args.arkane_output):
        dirs.sort()
        for directory in dirs:
            index = directory[3:]
            indices.append(index)
        break
    print(f'\nNumber of reactions: {len(indices)}')

    df_cleaned = pd.read_csv(args.cleaned_csv)

    output = Parallel(n_jobs=args.n_cpus)(delayed(parse_rates)(args, index, df_cleaned) for index in indices)
    cols = ['idx', 'rsmi', 'psmi'] 
    cols.extend([f'k(T{i})' for i in range(50)])
    cols.extend([f'k(T{i})_rev' for i in range(50)])
    df = pd.DataFrame(output, columns=cols)

    df.to_csv(args.out_name, index=False)


if __name__ == '__main__':
    main()
