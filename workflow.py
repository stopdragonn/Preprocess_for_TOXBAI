"""Reusable workflow functions for the SMILES preprocessing pipeline."""

from __future__ import annotations

from multiprocessing import Pool
from typing import Callable, Dict, Optional, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover, Descriptors
from tqdm import tqdm

# Allowed atoms for organic subset
ALLOWED_ATOMS = {
    'C', 'N', 'O', 'S', 'P', 'B',
    'F', 'Cl', 'Br', 'I',
    'H', 'D', 'T'
}

# Descriptor functions used when computing descriptors
DESCRIPTOR_FUNCS: Dict[str, Callable[[Chem.Mol], Any]] = {
    'SlogP': Descriptors.MolLogP,
    'SMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
    'MolWt': Descriptors.MolWt,
    'NumHAcceptors': Descriptors.NumHAcceptors,
    'NumHDonors': Descriptors.NumHDonors,
    'TPSA': Descriptors.TPSA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
}


def load_salt_remover(defn_file: str) -> SaltRemover.SaltRemover:
    """Load RDKit SaltRemover from definition file."""
    return SaltRemover.SaltRemover(defnFile=defn_file, onlyUncharged=False)


def strip_salts(smiles: str, remover: SaltRemover.SaltRemover) -> Optional[str]:
    """Remove salts from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(stripped)


def filter_organic(smiles: str) -> bool:
    """Return True if molecule contains only allowed organic atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset(ALLOWED_ATOMS)


def calc_descriptors(smiles: str) -> Dict[str, Optional[float]]:
    """Calculate a set of RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {name: None for name in DESCRIPTOR_FUNCS}
    values: Dict[str, Optional[float]] = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            values[name] = func(mol)
        except Exception:
            values[name] = None
    return values


def parallel_series_apply(
    series: pd.Series,
    func: Callable[[str], Any],
    n_procs: int = 1,
    desc: Optional[str] = None,
    chunksize: int = 100,
) -> pd.Series:
    """Apply a function to a pandas Series with optional multiprocessing.

    Parameters
    ----------
    series : pd.Series
        Input data.
    func : Callable[[str], Any]
        Function to apply to each element.
    n_procs : int, optional
        Number of processes to use. Defaults to ``1`` (sequential).
    desc : Optional[str], optional
        Description for the progress bar.
    chunksize : int, optional
        Chunk size for ``multiprocessing.Pool.imap``.

    Returns
    -------
    pd.Series
        Series of results matching the index of ``series``.
    """
    if n_procs <= 1:
        # Sequential processing with a progress bar
        results = []
        for item in tqdm(series, desc=desc):
            results.append(func(item))
        return pd.Series(results, index=series.index)

    with Pool(processes=n_procs) as pool:
        iterator = pool.imap(func, series.tolist(), chunksize)
        results = list(tqdm(iterator, total=len(series), desc=desc))
    return pd.Series(results, index=series.index)
