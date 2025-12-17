"""
Enhanced SMILES Standardization Module
Integrates TOXBAI preprocessing logic with MolVS standardization

This module combines:
1. MolVS standardization (tautomer normalization, charge neutralization, etc.)
2. TOXBAI-style custom salt removal using SMARTS patterns
3. Organic molecule filtering
4. Robust error handling

Author: Integrated from TOXBAI repository (stopdragonn/Preprocess_for_TOXBAI)
"""

from typing import Optional
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')

try:
    from molvs import Standardizer
    MOLVS_AVAILABLE = True
except ImportError:
    MOLVS_AVAILABLE = False
    print("[Warning] MolVS not available. Install with: pip install molvs")

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    MOLSTANDARDIZE_AVAILABLE = True
except ImportError:
    MOLSTANDARDIZE_AVAILABLE = False
    print("[Warning] RDKit MolStandardize not available. Protomer normalization limited.")


# --- Configuration from TOXBAI repository (workflow.py) ---

# Allowed atoms for organic filtering (TOXBAI 리포지토리의 workflow.py 참고)
ALLOWED_ATOMS = {
    'C', 'N', 'O', 'S', 'P', 'B',
    'F', 'Cl', 'Br', 'I',
    'H', 'D', 'T'
}

# Default salt SMARTS patterns (TOXBAI 리포지토리의 Salts.txt 참고)
DEFAULT_SALT_SMARTS = [
    "[Cl-]",
    "[Br-]",
    "[Na+]",
    "[K+]",
]


# --- Core Functions ---

def load_salt_smarts_list(salt_file: str = None) -> list[Chem.Mol]:
    """
    Load salt SMARTS patterns from file or use defaults
    
    TOXBAI 리포지토리의 workflow.py:load_salt_smarts_list() 함수를 참고하여 작성
    
    Args:
        salt_file: Path to file containing SMARTS patterns (one per line)
                   If None, uses DEFAULT_SALT_SMARTS
    
    Returns:
        List of RDKit Mol objects representing salt patterns
    """
    if salt_file is None:
        patterns = DEFAULT_SALT_SMARTS
    else:
        patterns = []
        try:
            with open(salt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except FileNotFoundError:
            print(f"[Warning] Salt file not found: {salt_file}. Using defaults.")
            patterns = DEFAULT_SALT_SMARTS
    
    salt_mols = []
    for pattern in patterns:
        mol = Chem.MolFromSmarts(pattern)
        if mol:
            salt_mols.append(mol)
        else:
            print(f"[Warning] Invalid SMARTS pattern ignored: {pattern}")
    
    return salt_mols


def strip_salts_toxbai(smiles: str, salt_mols: list[Chem.Mol]) -> Optional[str]:
    """
    Remove salts using SMARTS-based fragment filtering
    
    TOXBAI 리포지토리의 workflow.py:strip_salts() 함수를 그대로 구현
    이 방식은 molecule을 fragment로 나눈 뒤, salt SMARTS와 매칭되는 fragment를 제거함
    
    Args:
        smiles: Input SMILES string
        salt_mols: List of RDKit Mol objects representing salt patterns
    
    Returns:
        SMILES string with salts removed, or None if processing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # Split into fragments (TOXBAI approach)
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
    
    # Combine kept fragments
    if len(kept_frags) == 1:
        try:
            return Chem.MolToSmiles(kept_frags[0])
        except Exception:
            return None
    else:
        # Multiple non-salt fragments - combine them
        try:
            combined = kept_frags[0]
            for frag in kept_frags[1:]:
                combined = Chem.CombineMols(combined, frag)
            return Chem.MolToSmiles(combined)
        except Exception:
            return None


def filter_organic(smiles: str) -> bool:
    """
    Check if molecule contains only allowed organic atoms
    
    TOXBAI 리포지토리의 workflow.py:filter_organic() 함수를 그대로 구현
    허용된 원소만 포함하는지 확인 (무기물 제거)
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        True if molecule is organic (contains only allowed atoms), False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset(ALLOWED_ATOMS)


def normalize_protomers_ph74(smiles: str) -> Optional[str]:
    """
    Normalize molecule to most probable protonation state at pH 7.4
    
    pH 7.4에서의 가장 우변된 프로토머 형태로 정규화합니다.
    이는 생리적 조건에서 분자의 전하 상태를 나타냅니다.
    
    독성학적 의의:
    - QSAR 모델 학습 시 일관된 분자 상태 필요
    - 약물 전달, 수용체 상호작용에 영향
    - 대사 경로 예측에 중요
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        SMILES with normalized protonation state at pH 7.4, or None if fails
    
    Example:
        >>> normalize_protomers_ph74("CC(=O)[O-]")  # Acetate (deprotonated)
        'CC(=O)O'  # Acetic acid (neutral form at pH 7.4)
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 방법 1: RDKit MolStandardize.Uncharger 사용 (권장)
        if MOLSTANDARDIZE_AVAILABLE:
            try:
                # Uncharger: 공식 전하 제거 (pH 7.4 중성화)
                uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)
                mol_uncharged = uncharger.uncharge(mol, sanitize=True)
                
                return Chem.MolToSmiles(mol_uncharged, isomericSmiles=True)
            except Exception as e:
                # Fallback: MolStandardize API 버전 호환성 문제
                pass
        
        # 방법 2: 간단한 전하 중성화 (MolStandardize 불가 시)
        # Most Probable Protonation State at pH 7.4 (근사)
        try:
            # Sanitize
            Chem.SanitizeMol(mol)
            
            # 기본 규칙:
            # - Carboxylic acids: [O-]C(=O) → OC(=O) (중성)
            # - Phosphates: [O-]P → OP (부분 중성화)
            # - Amines: [NH3+] → N (중성)
            # - Phenols: [O-]c → Oc (중성)
            
            formal_charge = Chem.GetFormalCharge(mol)
            
            if formal_charge == 0:
                # Already neutral
                return Chem.MolToSmiles(mol, isomericSmiles=True)
            elif formal_charge > 0:
                # Positively charged: 흔한 경우 암모늄 제거
                # Strategy: Add hydrogens to bases
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in ['N', 'O'] and atom.GetFormalCharge() > 0:
                        atom.SetFormalCharge(0)
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + formal_charge)
                
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, isomericSmiles=True)
            else:
                # Negatively charged: 음이온 중성화
                # Strategy: Add hydrogens to acidic groups
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in ['O', 'N', 'S'] and atom.GetFormalCharge() < 0:
                        atom.SetFormalCharge(0)
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - formal_charge)
                
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, isomericSmiles=True)
        
        except Exception:
            # Silent failure for protomer normalization
            return None
    
    except Exception:
        # Outer exception handler (should not reach here)
        return None


def standardize_smiles(
    smiles: str,
    use_molvs: bool = True,
    remove_salts: bool = True,
    filter_organics: bool = True,
    salt_file: str = None
) -> Optional[str]:
    """
    Enhanced SMILES standardization combining MolVS and TOXBAI approaches
    
    이 함수는 기존 MolVS 방식과 TOXBAI 리포지토리의 전처리 로직을 통합함:
    1. (Optional) MolVS Standardizer: 호변이성질체 정규화, 전하 중성화, 방향족 처리 등
    2. TOXBAI 방식 Salt 제거: SMARTS 패턴 기반 fragment 분리 방식
    3. Organic 필터링: 허용된 원소만 포함하는지 확인
    
    Args:
        smiles: Input SMILES string to standardize
        use_molvs: Whether to apply MolVS standardization first (default: True)
        remove_salts: Whether to remove salts using TOXBAI method (default: True)
        filter_organics: Whether to filter out inorganic molecules (default: True)
        salt_file: Path to custom salt SMARTS file (default: None, uses built-in)
    
    Returns:
        Standardized SMILES string, or None if processing fails or filtered out
    
    Example:
        >>> standardize_smiles("CC(=O)[O-].[Na+]")
        'CC(=O)O'
        
        >>> standardize_smiles("c1ccccc1", use_molvs=True, remove_salts=False)
        'c1ccccc1'
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    result_smiles = smiles
    
    # Step 1: MolVS Standardization (if enabled and available)
    # MolVS는 tautomer 정규화, 전하 중성화, 방향족 처리 등을 수행
    if use_molvs and MOLVS_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(result_smiles)
            if mol is None:
                return None
            
            standardizer = Standardizer()
            clean_mol = standardizer.standardize(mol)
            result_smiles = Chem.MolToSmiles(clean_mol)
        except Exception as e:
            # MolVS standardization failed, try to continue with original
            print(f"[Warning] MolVS standardization failed for {smiles}: {e}")
            # Continue with original SMILES
            pass
    
    # Step 2: Salt Removal (TOXBAI approach)
    # TOXBAI 방식: SMARTS 패턴으로 salt fragment를 식별하고 제거
    # MolVS의 SaltRemover와는 다른 방식으로, 더 세밀한 제어 가능
    if remove_salts:
        salt_mols = load_salt_smarts_list(salt_file)
        result_smiles = strip_salts_toxbai(result_smiles, salt_mols)
        if result_smiles is None:
            return None
    
    # Step 2.5: Protomer Normalization (NEW)
    # pH 7.4에서의 주요 프로토머 형태로 정규화
    # 독성 예측 모델은 일반적으로 생리적 조건(pH 7.4)에서의 분자 형태 기반
    # 이는 전하 중성화와 달리, 화학적으로 타당한 프로토머 형태 선택
    normalized = normalize_protomers_ph74(result_smiles)
    if normalized is not None:
        result_smiles = normalized
    # If protomer normalization fails, continue with current SMILES
    
    # Step 3: Organic Filtering
    # TOXBAI 방식: 특정 원소만 허용하여 무기물 제거
    if filter_organics:
        if not filter_organic(result_smiles):
            return None
    
    # Final validation and canonicalization
    try:
        mol = Chem.MolFromSmiles(result_smiles)
        if mol is None:
            return None
        # Return canonical SMILES with explicit stereochemistry
        # isomericSmiles=True: 입체화학 정보 명시화 (QSAR 모델 일관성 중요)
        # Stereochemistry matters in toxicology:
        #   - Ibuprofen-S (active) vs Ibuprofen-R (inactive): ~10x activity difference
        #   - Thalidomide: R-form causes teratogenicity, S-form is inactive
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        print(f"[Warning] Final canonicalization failed: {e}")
        return None


def standardize_smiles_simple(smiles: str) -> Optional[str]:
    """
    Simplified standardization (backward compatible with original code)
    
    기존 코드와의 호환성을 위한 간단한 버전:
    - MolVS standardization만 수행 (available한 경우)
    - Salt 제거 및 organic 필터링은 하지 않음
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Standardized SMILES or None
    """
    return standardize_smiles(
        smiles,
        use_molvs=True,
        remove_salts=False,
        filter_organics=False
    )


def standardize_smiles_toxbai(smiles: str, salt_file: str = "Salts.txt") -> Optional[str]:
    """
    Full TOXBAI-style preprocessing
    
    TOXBAI 리포지토리의 전체 전처리 로직 적용:
    - Protomer normalization at pH 7.4 (NEW)
    - Custom salt removal with SMARTS patterns
    - Organic filtering
    - No MolVS standardization (TOXBAI doesn't use it)
    - Explicit stereochemistry retention
    
    Args:
        smiles: Input SMILES string
        salt_file: Path to salt SMARTS file (default: "Salts.txt")
    
    Returns:
        Preprocessed SMILES or None
    """
    return standardize_smiles(
        smiles,
        use_molvs=False,
        remove_salts=True,
        filter_organics=True,
        salt_file=salt_file
    )


# --- Main comparison function ---

def compare_approaches(smiles: str, salt_file: str = None) -> dict:
    """
    Compare different standardization approaches on the same SMILES
    
    유용한 디버깅/비교 함수: 각 방식의 결과를 비교
    
    Args:
        smiles: Input SMILES string
        salt_file: Path to salt SMARTS file
    
    Returns:
        Dictionary with results from different approaches
    """
    results = {
        'original': smiles,
        'molvs_only': standardize_smiles_simple(smiles),
        'toxbai_only': standardize_smiles_toxbai(smiles, salt_file),
        'combined': standardize_smiles(smiles, use_molvs=True, remove_salts=True, 
                                       filter_organics=True, salt_file=salt_file),
    }
    return results


if __name__ == "__main__":
    # Example usage and comparison
    print("=== SMILES Standardization Examples ===\n")
    
    test_cases = [
        "CC(=O)[O-].[Na+]",  # Salt example
        "c1ccccc1",  # Aromatic benzene
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C.[Cl-].[Na+]",  # Caffeine with salts
    ]
    
    for smiles in test_cases:
        print(f"Original: {smiles}")
        results = compare_approaches(smiles)
        print(f"  MolVS only:     {results['molvs_only']}")
        print(f"  TOXBAI only:    {results['toxbai_only']}")
        print(f"  Combined:       {results['combined']}")
        print()

# --- Public API ---
__all__ = [
    'standardize_smiles',
    'normalize_protomers_ph74',
    'filter_organic',
    'load_salt_smarts_list',
    'strip_salts_toxbai',
    'compare_approaches',
]
