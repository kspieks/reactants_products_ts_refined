import argparse
from collections import Counter
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from arkane.common import symbol_by_number
from arkane.ess.factory import ess_factory
from arkane.encorr.corr import get_atom_correction, assign_frequency_scale_factor, get_bac
from arkane.modelchem import LevelOfTheory, CompositeLevelOfTheory

from rmgpy.molecule.molecule import Molecule
from rmgpy.quantity import ScalarQuantity
from rmgpy.statmech import HarmonicOscillator, IdealGasTranslation, LinearRotor, NonlinearRotor, Conformer


def log_to_enthalpy(smiles, freq_lot, energy_lot, freq_log_path, energy_log_path, temp=298.15):
    """
    Uses the Arkane API to calculate enthalpy from a QM log file

    Args:
        smiles: smiles string for the species
        freq_lot: LevelOfTheory object that specifies the LoT used for frequency calculation
        energy_lot: LevelOfTheory object that specifies the LoT used for energy calculation
        freq_log_path: path to the frequency log file
        energy_log_path: path to the energy log file
        temp: temperature to calculate enthalpy at

    Returns:
        enthalpy
    """
    freq_log = ess_factory(freq_log_path)    
    energy_log = ess_factory(energy_log_path)
    conformer, _ = freq_log.load_conformer()
    # perform quick checks
    assert conformer.spin_multiplicity > 0
    assert any(isinstance(mode, IdealGasTranslation) for mode in conformer.modes)
    assert any(isinstance(mode, (LinearRotor, NonlinearRotor)) for mode in conformer.modes)
    assert any(isinstance(mode, HarmonicOscillator) for mode in conformer.modes)
    
    # pass coords, nums, masses through function
    coords, nums, mass = freq_log.load_geometry()
    assert len(nums) > 1
    
    atoms = Counter(symbol_by_number[int(n)] for n in nums)
    conformer.coordinates = (coords, 'angstroms')
    conformer.number = nums
    conformer.mass = (mass, 'amu')
    
    freq_scale_factor = assign_frequency_scale_factor(freq_lot)
    frequencies = conformer.modes[2].frequencies.value_si
    for mode in conformer.modes:
        if isinstance(mode, HarmonicOscillator):
            mode.frequencies = (frequencies * freq_scale_factor, "cm^-1")
    if freq_scale_factor == 1:
        print('WARNING: Frequency scale factor is 1')
    zpe_scale_factor = freq_scale_factor / 1.014
    
    energy = energy_log.load_energy(zpe_scale_factor=zpe_scale_factor)  # J/mol
    
    # either supply zpe to function or supply list of freqs to function and calculate zpe here
    zpe = freq_log.load_zero_point_energy() * zpe_scale_factor if len(nums) > 1 else 0  # J/mol
    energy += zpe
    energy += get_atom_correction(energy_lot, atoms)  # J/mol
    bonds = Molecule().from_smiles(smiles).enumerate_bonds()
    bond_corrections = get_bac(energy_lot, bonds, coords, nums,
                               bac_type='p',
                               multiplicity=conformer.spin_multiplicity)
    energy += bond_corrections
    conformer.E0 = (energy / 4184, 'kcal/mol')
    
    return ScalarQuantity((conformer.get_enthalpy(temp) + conformer.E0.value_si) / 4184, 'kcal/mol')


def calculate_enthalpy(row, freq_lot, energy_lot, args):
    """
    Obtains ΔH for each reaction

    Args:
        row: one row from the dataframe
        args: command line args
    
    Returns:
        data: reaction enthalpies
    """
    data = []
    idx = int(row.idx)
    data.append(idx)
    data.append(row.rsmi)
    data.append(row.psmi)

    # get H298 for the reactant
    freq_log_path = os.path.join(args.opt_freq_dir, f'rxn{idx:06}', f'r{idx:06}.log')
    energy_log_path = os.path.join(args.sp_dir, f'rxn{idx:06}', f'r{idx:06}.log')
    H298_r = log_to_enthalpy(row.rsmi, freq_lot, energy_lot, freq_log_path, energy_log_path, temp=args.temp)
    data.append(H298_r.value)  # kcal/mol

    # get H298 for the product/s
    products = row.psmi.split('.')
    H298_p = []
    for i, product in enumerate(products):
        if len(products) == 1:
            freq_log_path = os.path.join(args.opt_freq_dir, f'rxn{idx:06}', f'p{idx:06}.log')
            energy_log_path = os.path.join(args.sp_dir, f'rxn{idx:06}', f'p{idx:06}.log')
        else:
            freq_log_path = os.path.join(args.opt_freq_dir, f'rxn{idx:06}', f'p{idx:06}_{i}.log')
            energy_log_path = os.path.join(args.sp_dir, f'rxn{idx:06}', f'p{idx:06}_{i}.log')
        
        enthalpy = log_to_enthalpy(product, freq_lot, energy_lot, freq_log_path, energy_log_path, temp=args.temp)
        H298_p.append(enthalpy.value)  # kcal/mol
    while len(H298_p) < 3:
        H298_p.append(np.NaN)
    data.extend(H298_p)

    # get ΔH298 for the reaction in kcal/mol
    H298_rxn = np.sum([p for p in H298_p if p is not np.NaN]) - np.sum(H298_r.value)
    data.append(float(H298_rxn))

    return np.array(data)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    Args:
        command_line_args: The command line arguments.
    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='Script for calculated enthalpies with atom and bond corrections')
    parser.add_argument('--csv', type=str, nargs=1, default='ccsdtf12_dz.csv',
                        help='csv file containing reaction index, rsmi, and psmi')
    parser.add_argument('--out_name', type=str, nargs=1, default='ccsdtf12_dz_enthalpies_corrected',
                        help='filename used for the output csv file')

    # QM level for geometry optimization + frequency calculation
    parser.add_argument('--method', type=str, nargs=1, default='wB97X-D3',
                        help='method used during the geometry optimization and frequency calculations')
    parser.add_argument('--basis', type=str, nargs=1, default='def2-TZVP',
                        help='basis used during the geometry optimization and frequency calculations')
    parser.add_argument('--software', type=str, nargs=1,
                        default='QChem',
                        help='software used during the geometry optimization and frequency calculations')
    parser.add_argument('--opt_freq_dir', type=str, nargs=1, default='wb97xd3/qm_logs',
                        help='directory containing log files from the quantum calculations')

    # QM level for single point calculation. If not specified, assumed to be the same as above
    parser.add_argument('--sp_method', type=str, nargs=1, default='ccsd(t)f12',
                        help='method used during the single point calculations')
    parser.add_argument('--sp_basis', type=str, nargs=1, default='cc-pvdz-f12',
                        help='basis used during the single point calculations')
    parser.add_argument('--sp_software', type=str, nargs=1,default='molpro',
                        help='software used during the single point calculations')
    parser.add_argument('--sp_dir', type=str, nargs=1, default='ccsdtf12/qm_logs',
                        help='directory containing log files from the quantum calculations')

    # specify temperature
    parser.add_argument('--temp', type=float, nargs=1, default=298.15,
                        help='temperature in Kelvin to use while calculating the entropy')

    args = parser.parse_args(command_line_args)

    if (args.sp_method is None) and (args.sp_basis is None) and (args.sp_software is None) and (args.sp_dir is None):
        args.sp_method = args.method
        args.sp_basis = args.basis
        args.sp_software = args.software
        args.sp_dir = args.opt_freq_dir

    return args


def main():
    args = parse_command_line_arguments()
    print('Using arguments...')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    freq_lot = LevelOfTheory(method=args.method, basis=args.basis, software=args.software)
    energy_lot = LevelOfTheory(method=args.sp_method, basis=args.sp_basis, software=args.sp_software)
    if args.method != args.sp_method and args.basis != args.sp_basis:
        energy_lot = CompositeLevelOfTheory(freq=freq_lot, energy=energy_lot)

    # count number of reaction directories
    num_rxns = len(os.listdir(args.sp_dir))
    print(f'\nNumber of reactions: {num_rxns}')

    df = pd.read_csv(args.csv)
    cols = ['idx', 'rsmi', 'psmi', 'H298_r', 'H298_p0', 'H298_p1', 'H298_p2', 'dHrxn298']
    df_H298 = pd.DataFrame(np.zeros((0, len(cols))), columns=cols)
    
    rows = list(df.iterrows())
    for i, row in rows:
        data = calculate_enthalpy(row, freq_lot, energy_lot, args)
        df_H298 = df_H298.append(pd.DataFrame(data.reshape(1, len(cols)), columns=cols), ignore_index=True)

    # convert type to float
    df_H298.H298_r = df_H298.H298_r.astype(float)
    df_H298.H298_p0 = df_H298.H298_p0.astype(float)
    df_H298.H298_p1 = df_H298.H298_p1.astype(float)
    df_H298.H298_p2 = df_H298.H298_p2.astype(float)
    df_H298.dHrxn298 = df_H298.dHrxn298.astype(float)

    df_H298.to_csv(f'{args.out_name}.csv', index=False)
    print(df_H298.info())


if __name__ == '__main__':
    main()
