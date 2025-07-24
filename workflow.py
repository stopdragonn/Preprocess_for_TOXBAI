from rdkit import Chem
from rdkit.Chem import Descriptors
from multiprocessing import Pool
from typing import Callable, Optional
import pandas as pd
from tqdm import tqdm

# --- 설정 ---

ALLOWED_ATOMS = {
    'C', 'N', 'O', 'S', 'P', 'B',
    'F', 'Cl', 'Br', 'I',
    'H', 'D', 'T'
}

DESCRIPTOR_FUNCS = {
    'SlogP': Descriptors.MolLogP,
    'SMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
}

# --- salt SMARTS 수동 로딩 및 제거 로직 ---

def load_salt_smarts_list(salt_file: str) -> list[Chem.Mol]:
    """Salts.txt 파일에서 SMARTS 리스트 로드 (RDKit 최신 버전 호환)"""
    salts = []
    with open(salt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mol = Chem.MolFromSmarts(line)
            if mol:
                salts.append(mol)
            else:
                print(f"[경고] 잘못된 SMARTS 무시됨: {line}")
    return salts


def strip_salts(smiles: str, salt_mols: list[Chem.Mol]) -> Optional[str]:
    """SMARTS 기반 salt 제거 구현 (가장 단순한 fragment 분리 방식)"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    kept_frags = []

    for frag in fragments:
        is_salt = False
        for salt in salt_mols:
            if frag.HasSubstructMatch(salt):
                is_salt = True
                break
        if not is_salt:
            kept_frags.append(frag)

    if not kept_frags:
        return None

    if len(kept_frags) == 1:
        return Chem.MolToSmiles(kept_frags[0])
    else:
        combined = kept_frags[0]
        for frag in kept_frags[1:]:
            combined = Chem.CombineMols(combined, frag)
        return Chem.MolToSmiles(combined)


# --- 유기 필터, descriptor 계산 ---

def filter_organic(smiles: str) -> bool:
    """허용된 원소만 포함된 유기 화합물인지 확인"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset(ALLOWED_ATOMS)


def calc_descriptors(smiles: str) -> dict:
    """RDKit descriptor 계산"""
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


# --- 병렬 적용 유틸리티 ---

def parallel_series_apply(
    series: pd.Series,
    func: Callable[[str], object],
    n_procs: int = 1,
    desc: Optional[str] = None,
    chunksize: int = 100,
) -> pd.Series:
    """병렬 또는 순차적으로 시리즈에 함수 적용"""
    if n_procs <= 1:
        results = []
        for item in tqdm(series, desc=desc):
            results.append(func(item))
        return pd.Series(results, index=series.index)

    with Pool(processes=n_procs) as pool:
        iterator = pool.imap(func, series.tolist(), chunksize)
        results = list(tqdm(iterator, total=len(series), desc=desc))
    return pd.Series(results, index=series.index)
