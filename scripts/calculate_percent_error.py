"""
Calculates the percent error between the fitted rate constants and TST calculation.
"""

import argparse
import numpy as np
import pandas as pd

def calculate_percent_error(df, temps):
    """
    Calculate the average percent error between the rate constants 
    calculated from TST and those predicted using the fitted Arrhenius parameters.

    Args:
        df: dataframe containing all results for the respective level of theory
        temps: dataframe containing the list of temperatures (K) used for fitting
    
    Returns:
        df: the original dataframe with a new column containing the percent error
    """
    R = 1.98720425864083e-3  # kcal/mol/K
    percent_errors = np.zeros(len(df))
    percent_errors[:] = np.nan
    for i, idx in enumerate(df.idx.values):
        lnA = df[df.idx == idx].lnA.values[0]
        Ea = df[df.idx == idx].Ea.values[0]
        fitted_rates = np.exp(lnA) * np.exp(-Ea/R /temps.values.flatten())
        
        # 50 temperatures were used for this fitting
        calculated_rates = df[df.idx == idx].iloc[:, 7:7+50].values.flatten()
        percent_err = np.abs( (calculated_rates - fitted_rates) / calculated_rates * 100 )
        percent_errors[i] = np.mean(percent_err)
        
    df['percent_error'] = percent_errors

    return df


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for calculating the percent error between the fitted and calculated rate constants')
    parser.add_argument('--ccsd', type=str, nargs=1, 
                        default='ccsdtf12/ccsdtf12_dz.csv',
                        help='path to csv containing ccsd results')
    parser.add_argument('--wb97xd3', type=str, nargs=1,  
                        default='wb97xd3/wb97xd3.csv',
                        help='path to csv containing wb97xd3 results')
    parser.add_argument('--b97d3', type=str, nargs=1,
                        default='b97d3/b97d3.csv',
                        help='path to csv containing wb97xd3 results')
    parser.add_argument('--temps', type=str, nargs=1,
                        default='arkane_tempeatures.csv',
                        help='path to csv containing temperatures used during Arrhenius fitting')
  
    args = parser.parse_args(command_line_args)

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    temps = pd.read_csv(args.temps, index_col=0)

    df_ccsd = pd.read_csv(args.ccsd)
    df_ccsd = calculate_percent_error(df_ccsd, temps)
    df_ccsd.to_csv('ccsdtf12_dz_percent_errors.csv', index=False)

    df_wb97xd3 = pd.read_csv(args.wb97xd3)
    df_wb97xd3 = calculate_percent_error(df_wb97xd3, temps)
    df_wb97xd3.to_csv('wb97xd3_percent_errors.csv', index=False)

    df_b97d3 = pd.read_csv(args.b97d3)
    df_b97d3 = calculate_percent_error(df_b97d3, temps)
    df_b97d3.to_csv('b97d3_percent_errors.csv', index=False)


if __name__ == '__main__':
    main()