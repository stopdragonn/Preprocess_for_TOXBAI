from rdkit import Chem
from rdkit.Chem import SaltRemover, Descriptors
from multiprocessing import Pool
from typing import Callable
import pandas as pd
from tqdm import tqdm

# Allowed atoms for organic subset
ALLOWED_ATOMS = {
    'C', 'N', 'O', 'S', 'P', 'B',
    'F', 'Cl', 'Br', 'I',
    'H', 'D', 'T'
}

# Descriptor functions used when computing descriptors
DESCRIPTOR_FUNCS = {
    'SlogP': Descriptors.MolLogP,
    'SMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
}


def load_salt_remover(defn_file: str) -> SaltRemover.SaltRemover:
    """Load RDKit SaltRemover from definition file."""
    return SaltRemover.SaltRemover(defnFile=defn_file, onlyUncharged=False)


def strip_salts(smiles: str, remover: SaltRemover.SaltRemover) -> str:
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


def calc_descriptors(smiles: str) -> dict:
    """Calculate a set of RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {name: None for name in DESCRIPTOR_FUNCS}
    values = {}
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            values[name] = func(mol)
        except Exception:
            values[name] = None
    return values


def parallel_series_apply(
    series: pd.Series,
    func: Callable[[str], object],
    n_procs: int = 1,
    desc: str | None = None,
    chunksize: int = 100,
) -> pd.Series:
    """Apply a function to a pandas Series with optional multiprocessing."""
    if n_procs <= 1:
        return series.progress_apply(func, desc=desc)

    with Pool(processes=n_procs) as pool:
        iterator = pool.imap(func, series.tolist(), chunksize)
        results = list(tqdm(iterator, total=len(series), desc=desc))
    return pd.Series(results, index=series.index)
