from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
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
    # Basic descriptors
    'SlogP': Descriptors.MolLogP,
    'SMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
    'TPSA': Descriptors.TPSA,
    'AMW': Descriptors.MolWt,
    'ExactMW': Descriptors.ExactMolWt,
    'NumLipinskiHBA': Descriptors.NumHAcceptors,
    'NumLipinskiHBD': Descriptors.NumHDonors,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumHBD': Descriptors.NumHDonors,
    'NumHBA': Descriptors.NumHAcceptors,
    'NumAmideBonds': Descriptors.NumAmideBonds,
    'NumHeteroAtoms': Descriptors.NumHeteroatoms,
    'NumHeavyAtoms': Descriptors.HeavyAtomCount,
    'NumAtoms': lambda mol: mol.GetNumAtoms(),
    'NumStereocenters': Descriptors.NumAtomStereoCenters,
    'NumUnspecifiedStereocenters': Descriptors.NumUnspecifiedAtomStereoCenters,
    'NumRings': Descriptors.RingCount,
    'NumAromaticRings': Descriptors.NumAromaticRings,
    'NumSaturatedRings': Descriptors.NumSaturatedRings,
    'NumAliphaticRings': Descriptors.NumAliphaticRings,
    'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
    'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles,
    'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
    'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
    'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles,
    'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
    'FractionCSP3': Descriptors.FractionCSP3,
    
    # Chi connectivity indices
    'Chi0v': Descriptors.Chi0v,
    'Chi1v': Descriptors.Chi1v,
    'Chi2v': Descriptors.Chi2v,
    'Chi3v': Descriptors.Chi3v,
    'Chi4v': Descriptors.Chi4v,
    'Chi1n': Descriptors.Chi1n,
    'Chi2n': Descriptors.Chi2n,
    'Chi3n': Descriptors.Chi3n,
    'Chi4n': Descriptors.Chi4n,
    
    # Kappa descriptors
    'HallKierAlpha': Descriptors.HallKierAlpha,
    'kappa1': Descriptors.Kappa1,
    'kappa2': Descriptors.Kappa2,
    'kappa3': Descriptors.Kappa3,
    
    # VSA descriptors - SlogP_VSA
    'slogp_VSA1': Descriptors.SlogP_VSA1,
    'slogp_VSA2': Descriptors.SlogP_VSA2,
    'slogp_VSA3': Descriptors.SlogP_VSA3,
    'slogp_VSA4': Descriptors.SlogP_VSA4,
    'slogp_VSA5': Descriptors.SlogP_VSA5,
    'slogp_VSA6': Descriptors.SlogP_VSA6,
    'slogp_VSA7': Descriptors.SlogP_VSA7,
    'slogp_VSA8': Descriptors.SlogP_VSA8,
    'slogp_VSA9': Descriptors.SlogP_VSA9,
    'slogp_VSA10': Descriptors.SlogP_VSA10,
    'slogp_VSA11': Descriptors.SlogP_VSA11,
    'slogp_VSA12': Descriptors.SlogP_VSA12,
    
    # VSA descriptors - SMR_VSA
    'smr_VSA1': Descriptors.SMR_VSA1,
    'smr_VSA2': Descriptors.SMR_VSA2,
    'smr_VSA3': Descriptors.SMR_VSA3,
    'smr_VSA4': Descriptors.SMR_VSA4,
    'smr_VSA5': Descriptors.SMR_VSA5,
    'smr_VSA6': Descriptors.SMR_VSA6,
    'smr_VSA7': Descriptors.SMR_VSA7,
    'smr_VSA8': Descriptors.SMR_VSA8,
    'smr_VSA9': Descriptors.SMR_VSA9,
    'smr_VSA10': Descriptors.SMR_VSA10,
    
    # VSA descriptors - PEOE_VSA
    'peoe_VSA1': Descriptors.PEOE_VSA1,
    'peoe_VSA2': Descriptors.PEOE_VSA2,
    'peoe_VSA3': Descriptors.PEOE_VSA3,
    'peoe_VSA4': Descriptors.PEOE_VSA4,
    'peoe_VSA5': Descriptors.PEOE_VSA5,
    'peoe_VSA6': Descriptors.PEOE_VSA6,
    'peoe_VSA7': Descriptors.PEOE_VSA7,
    'peoe_VSA8': Descriptors.PEOE_VSA8,
    'peoe_VSA9': Descriptors.PEOE_VSA9,
    'peoe_VSA10': Descriptors.PEOE_VSA10,
    'peoe_VSA11': Descriptors.PEOE_VSA11,
    'peoe_VSA12': Descriptors.PEOE_VSA12,
    'peoe_VSA13': Descriptors.PEOE_VSA13,
    'peoe_VSA14': Descriptors.PEOE_VSA14,
}

# MQN descriptors (handled separately due to array return)
def get_mqn_descriptors(mol):
    """Get MQN descriptors as individual named descriptors"""
    try:
        mqns = rdMolDescriptors.MQNs_(mol)
        return {f'MQN{i+1}': mqns[i] for i in range(min(42, len(mqns)))}
    except:
        return {f'MQN{i+1}': None for i in range(42)}

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
    """SMARTS 기반 salt 제거 구현 (분자 구조 유효성 검증 포함)"""
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

    # Combine fragments and validate the resulting molecule
    if len(kept_frags) == 1:
        result_mol = kept_frags[0]
    else:
        result_mol = kept_frags[0]
        for frag in kept_frags[1:]:
            result_mol = Chem.CombineMols(result_mol, frag)
    
    # Validate the resulting molecule structure
    try:
        # Try to sanitize the molecule to check if it's chemically valid
        Chem.SanitizeMol(result_mol)
        result_smiles = Chem.MolToSmiles(result_mol)
        
        # Additional validation: check if the SMILES can be parsed back
        validation_mol = Chem.MolFromSmiles(result_smiles)
        if validation_mol is None:
            return None
            
        return result_smiles
    except:
        # If sanitization fails, the molecule is not chemically valid
        return None


# --- 유기 필터, descriptor 계산 ---

def filter_organic(smiles: str) -> bool:
    """허용된 원소만 포함된 유기 화합물인지 확인"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset(ALLOWED_ATOMS)


def calc_descriptors(smiles: str) -> dict:
    """RDKit descriptor 계산 (확장된 descriptor 리스트 포함)"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        # Return None for all descriptors including MQNs
        values = {name: None for name in DESCRIPTOR_FUNCS}
        values.update({f'MQN{i+1}': None for i in range(42)})
        return values
    
    values = {}
    
    # Calculate standard descriptors
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            values[name] = func(mol)
        except Exception:
            values[name] = None
    
    # Calculate MQN descriptors
    mqn_values = get_mqn_descriptors(mol)
    values.update(mqn_values)
    
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
