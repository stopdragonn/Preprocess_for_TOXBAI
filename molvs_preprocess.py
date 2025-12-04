from rdkit import Chem
from molvs import Standardizer
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Uncharger

def preprocess_pipeline(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 1. [Standardizer] 기본 정규화 (토토머, 이상한 결합 수정)
    s = Standardizer()
    mol = s.standardize(mol)
    
    # 2. [LargestFragmentChooser] 가장 큰 조각만 남기기 (★핵심: 염/용매 제거)
    lfc = LargestFragmentChooser()
    mol = lfc.choose(mol)
    
    # 3. [Uncharger] 전하 중화 (선택사항: COO-를 COOH로, NH3+를 NH2로)
    # 독성 모델은 보통 중성 상태를 선호하므로 추가하는 것이 좋습니다.
    u = Uncharger()
    mol = u.uncharge(mol)
    
    return Chem.MolToSmiles(mol)

# --- 테스트 데이터 ---
# 1. Metformin HCl (염산염이 붙어 있음)
# 2. Diclofenac Sodium (나트륨염, 전하 존재)
# 3. 이상한 믹스처 (물과 염이 섞임)
test_data = [
    "CN(C)C(=N)N=C(N)N.Cl",                  # Metformin HCl
    "[Na+].[O-]C(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", # Diclofenac Sodium
    "CCC.O.[Na+].[Cl-]"                      # Mixture
]

print(f"{'Original':<45} | {'Processed (Final)':<40}")
print("-" * 90)

for smi in test_data:
    clean_smi = preprocess_pipeline(smi)
    print(f"{smi:<45} | {clean_smi:<40}")
