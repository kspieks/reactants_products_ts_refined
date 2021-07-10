"""
This script runs Arkane to determine the rate constant from a quantum calculation.
Requires installation of
RMG-Py: https://github.com/ReactionMechanismGenerator/RMG-Py
RMG-database: https://github.com/ReactionMechanismGenerator/RMG-database
ARC: https://github.com/ReactionMechanismGenerator/ARC

Be sure to checkout the update_qchem_parser branch on RMG-Py 
and activate the arc_env before running this script.
"""

import argparse
import numpy as np
import os
import pandas as pd
import time

from arc.level import Level
from arc.parser import parse_xyz_from_file
from arc.species.species import ARCSpecies
from arc.reaction import ARCReaction
from arc.statmech.arkane import ArkaneAdapter

from arkane.modelchem import LevelOfTheory
from arkane.encorr.corr import assign_frequency_scale_factor


def get_product_from_qm_log(args, opt_log, sp_log, rxn_index, product_index):
    """
    Creates an ARCSpecies object of the product from the optimized 
    xyz coordinates in the QM log file. Supports either QChem or MOLPRO.

    Args:
        args: command line arguments
        p_log: filepath to the product QChem log file
        rxn_index: padded index for the reaction. Example: 000001
        product_index: One of 0, 1, or 2

    Returns:
        p: ARCSpecies object for the product
    """
    xyz = parse_xyz_from_file(opt_log)

    if args.sp_software.lower() == 'qchem':
        with open(sp_log, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '$molecule' in line:
                    line = lines[i+1]
                    charge = int(line[0])
                    multiplicity = int(line[2])
                    break
    elif args.sp_software.lower() == 'molpro':
        with open(sp_log, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'maxit' in line:
                    line = lines[i+1]
                    charge = int(line[9])
                    multiplicity = int(line[18]) + 1
                    break
    if charge != 0:
        print(f'ERROR: p{rxn_index}_{product_index} had charge + {charge}')
    if multiplicity != 1:
        print(f'ERROR: p{rxn_index}_{product_index} had multiplicity + {multiplicity}')
    p = ARCSpecies(label=f'p{rxn_index}_{product_index}', xyz=xyz, multiplicity=multiplicity, charge=charge)

    return p


def run_arkane(args):
    """
    Function for obtaining rate constants using Arkane.

    Args:
        args: command line arguments
    """
    
    # confirm the number of reactions
    for _, dirs, files in os.walk(args.sp_dir):
        dirs.sort()
        print(f'{args.sp_dir} contains {np.unique(dirs).shape[0]} reactions')
        break

    # level of theory used during optimization and frequency calculation
    LOT = LevelOfTheory(method=args.method, basis=args.basis, software=args.software)
    freq_scale_factor = assign_frequency_scale_factor(LOT)
    print(f'\nUsing frequency scale factor {freq_scale_factor} for {LOT.to_model_chem()}...')

    # level of theory using during single point calculation
    sp_level = Level(method=args.sp_method, basis=args.sp_basis, software=args.sp_software)
    print(f'Using {sp_level} for single point energy...\n')

    # define kinetic settings for fitting k(T)
    T_min = (args.T_min, 'K')
    T_max = (args.T_max, 'K')
    T_count = args.T_count
    three_params = False

    total = 0

    start = time.time()
    for root, dirs, files in os.walk(args.sp_dir):
        dirs.sort()
        for i, directory in enumerate(dirs):
            print(f'*'*80)
            print(f'{directory}')
            # get the number of products
            for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
                num_files = len(files1)
                num_products = num_files - 2
                break

            total += 1
            index = directory[3:]
            output_directory = os.path.join(args.output_dir, 'rxn'+index)

            # define required input arguments 
            rxn = f'rxn{index}'
            r_log = f'r{index}.log'
            ts_log = f'ts{index}.log'
            if num_products > 1:
                p0_log = f'p{index}_0.log'
            else:
                p0_log = f'p{index}.log'

            product_logs = [p0_log]
            product_labels = [f'p{index}_0']

            # ArkaneAdapter only needs 'paths' and 'convergence'. everything else is not used by the adapter class
            output = {f'r{index}': {'paths': {'geo':  os.path.join(args.opt_freq_dir, rxn, r_log),
                                              'freq': os.path.join(args.opt_freq_dir, rxn, r_log),
                                              'sp':   os.path.join(args.sp_dir, rxn, r_log),
                                              'composite': None,
                                              },
                                    'convergence': True,
                                 },
                      f'p{index}_0': {'paths': {'geo':  os.path.join(args.opt_freq_dir, rxn, p0_log),
                                                'freq': os.path.join(args.opt_freq_dir, rxn, p0_log),
                                                'sp':   os.path.join(args.sp_dir, rxn, p0_log),
                                                'composite': None,
                                                },
                                      'convergence': True,
                                },
                      f'ts{index}': {'paths': {'geo':  os.path.join(args.opt_freq_dir, rxn, ts_log),
                                               'freq': os.path.join(args.opt_freq_dir, rxn, ts_log),
                                               'sp':   os.path.join(args.sp_dir, rxn, ts_log),
                                               'composite': None,
                                               },
                                     'convergence': True,
                            },
                      }
            if num_products > 1:
                p1_log = 'p' + index + '_1.log'
                product_logs.append(p1_log)
                product_labels.append(p1_log.split('.')[0])
                output.update({f'p{index}_1': {'paths': {'geo':  os.path.join(args.opt_freq_dir, rxn, p1_log),
                                                         'freq': os.path.join(args.opt_freq_dir, rxn, p1_log),
                                                         'sp':   os.path.join(args.sp_dir, rxn, p1_log),
                                                         'composite': None,
                                                         },
                                               'convergence': True,
                                               },
                                })
            if num_products > 2:
                p2_log = 'p' + index + '_2.log'
                product_logs.append(p2_log)
                product_labels.append(p2_log.split('.')[0])
                output.update({f'p{index}_2': {'paths': {'geo':  os.path.join(args.opt_freq_dir, rxn, p2_log),
                                                         'freq': os.path.join(args.opt_freq_dir, rxn, p2_log),
                                                         'sp':   os.path.join(args.sp_dir, rxn, p2_log),
                                                         'composite': None,
                                                         },
                                               'convergence': True,
                                                },
                                })
            
            # define ARCReaction
            reaction = ARCReaction(reactants=[f'r{index}'], products=product_labels)

            # define reactant
            xyz = parse_xyz_from_file(os.path.join(args.opt_freq_dir, rxn, r_log))
            r1 = ARCSpecies(label=f'r{index}', xyz=xyz, multiplicity=1, charge=0)

            # define ts
            reaction.ts_label = f'ts{index}'
            xyz = parse_xyz_from_file(os.path.join(args.opt_freq_dir, rxn, ts_log))
            ts = ARCSpecies(label=f'ts{index}', is_ts=True, xyz=xyz, multiplicity=1, charge=0)
            
            species_dict = {f'r{index}': r1, f'ts{index}': ts}

            # define products
            arc_products = []
            for i in range(num_products):
                p_log = product_logs[i]
                opt_log_path = os.path.join(args.opt_freq_dir, rxn, p_log)
                sp_log_path = os.path.join(args.sp_dir, rxn, p_log)
                p = get_product_from_qm_log(args, opt_log_path, sp_log_path, index, i)
                arc_products.append(p)
                species_dict.update({f'p{index}_{i}': p})

            # assign attributes
            reaction.r_species = [r1]
            reaction.ts_species = ts
            reaction.p_species = arc_products               
            
            # Run Arkane 
            # use code from line 112 in ARC/arc/processor.py
            # first get the thermo and store in Species/ folder
            for species in reaction.r_species + reaction.p_species:
                arkane = ArkaneAdapter(output_directory=output_directory,
                                       output_dict=output,
                                       bac_type=None,
                                       sp_level=sp_level,
                                       freq_scale_factor=freq_scale_factor,
                                       species=species,
                                      )
                arkane.compute_thermo(kinetics_flag=True)
            
            # next calculate kinetics and store in rxns/ folder
            arkane = ArkaneAdapter(output_directory=output_directory,
                                   output_dict=output,
                                   bac_type=None,
                                   sp_level=sp_level,
                                   freq_scale_factor=freq_scale_factor,
                                   reaction=reaction,
                                   species_dict=species_dict,
                                   T_min=T_min,
                                   T_max=T_max,
                                   T_count=T_count,
                                   three_params=three_params,
                                  )
            arkane.compute_high_p_rate_coefficient()
        break
    print(f'Total: {total}')
    print(f'Elapsed time: {time.time() - start:.2f} seconds')


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for running Arkane to obtain rates')
    # QM level for geometry optimization + frequency calculation
    parser.add_argument('--method', type=str, nargs=1, default='wB97X-D3',
                        help='method used during the geometry optimization and frequency calculations')
    parser.add_argument('--basis', type=str, nargs=1, default='def2-TZVP',
                        help='basis used during the geometry optimization and frequency calculations')
    parser.add_argument('--software', type=str, nargs=1, default='QChem',
                        help='software used during the geometry optimization and frequency calculations')
    parser.add_argument('--opt_freq_dir', type=str, nargs=1, default=None,
                        help='directory containing log files from the quantum calculations')

    # QM level for single point calculation. If not specified, assumed to be the same as above
    parser.add_argument('--sp_method', type=str, nargs=1, default=None,
                        help='method used during the single point calculations')
    parser.add_argument('--sp_basis', type=str, nargs=1, default=None,
                        help='basis used during the single point calculations')
    parser.add_argument('--sp_software', type=str, nargs=1, default=None,
                        help='software used during the single point calculations')
    parser.add_argument('--sp_dir', type=str, nargs=1, default=None,
                        help='directory containing log files from the quantum calculations')

    # specify output directory
    parser.add_argument('--output_dir', type=str, nargs=1, default=None,
                        help='directory to store the output files from Arkane')
    
    # specify temperatures for the fit
    parser.add_argument('--T_min', type=float, nargs=1, default=300,
                        help='minimum temperature in Kelvin to use during the fitting')
    parser.add_argument('--T_max', type=float, nargs=1, default=2000,
                        help='maximum temperature in Kelvin to use during the fitting')
    parser.add_argument('--T_count', type=float, nargs=1, default=50,
                        help='number of points between T_min and T_max')

    args = parser.parse_args(command_line_args)

    if (args.sp_method is None) and (args.sp_basis is None) and (args.sp_software is None):
        args.sp_method = args.method
        args.sp_basis = args.basis
        args.sp_software = args.software
        args.sp_dir = args.opt_freq_dir

    if args.output_dir is None:
        args.output_dir = f'{args.sp_method}_arkane_output'

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    run_arkane(args)


if __name__ == '__main__':
    main()
