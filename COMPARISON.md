# SMILES 전처리 방법 비교: MolVS vs TOXBAI

## 📊 주요 차이점 요약

| 항목 | MolVS Standardizer | TOXBAI Repository |
|------|-------------------|-------------------|
| **주요 목적** | 화학 구조 표준화 (tautomer, charge, aromaticity) | 독성 예측을 위한 데이터 전처리 |
| **Salt 제거 방식** | `SaltRemover` 클래스 (내장 salt list) | SMARTS 패턴 기반 fragment 필터링 |
| **Standardization** | 자동 (tautomer 정규화, 전하 중성화 등) | 없음 (salt 제거와 필터링만) |
| **Organic 필터링** | 없음 | 특정 원소만 허용 (C,N,O,S,P,B,F,Cl,Br,I,H,D,T) |
| **Canonicalization** | 자동 포함 | RDKit 기본 canonical SMILES |
| **에러 처리** | 예외 발생 가능 | 실패 시 None 반환 |

---

## 🔍 상세 분석

### 1. Salt 제거 방식의 차이

#### MolVS 방식:
```python
from rdkit.Chem import SaltRemover
remover = SaltRemover.SaltRemover()
stripped = remover.StripMol(mol)
```

- **특징**:
  - RDKit 내장 salt 리스트 사용
  - 가장 큰 fragment를 자동으로 선택
  - 간단하고 빠름

#### TOXBAI 방식:
```python
# workflow.py의 strip_salts() 함수 참고
fragments = Chem.GetMolFrags(mol, asMols=True)
for frag in fragments:
    if not frag.HasSubstructMatch(salt_pattern):
        kept_frags.append(frag)
```

- **특징**:
  - 커스텀 SMARTS 패턴 사용 (Salts.txt)
  - 각 fragment를 개별적으로 검사
  - Salt가 아닌 여러 fragment를 모두 유지 가능
  - 더 세밀한 제어 가능

**주요 차이**: 
- MolVS는 **가장 큰 fragment만** 선택
- TOXBAI는 **salt가 아닌 모든 fragment**를 유지

---

### 2. Standardization (표준화) 처리

#### MolVS Standardizer:
```python
from molvs import Standardizer
s = Standardizer()
clean_mol = s.standardize(mol)
```

**수행하는 작업**:
1. **Tautomer 정규화**: 호변이성질체를 canonical 형태로 변환
2. **전하 중성화**: 가능한 경우 중성 형태로 변환
3. **방향족 처리**: 방향족 고리의 표현 통일
4. **Stereochemistry 정리**: 입체화학 정보 정리
5. **Fragment 선택**: 가장 큰 fragment 선택

#### TOXBAI 방식:
- **명시적인 standardization 없음**
- Salt 제거와 organic 필터링만 수행
- RDKit의 기본 `MolToSmiles()`를 통한 canonical화만 수행

**선택 기준**:
- **MolVS 사용**: 화학 구조의 일관성이 중요한 경우 (QSAR 모델링, 중복 제거)
- **TOXBAI 방식**: 원본 구조 보존이 중요한 경우 (독성 예측 데이터셋)

---

### 3. Organic 필터링

#### TOXBAI의 Organic 필터:
```python
# workflow.py의 ALLOWED_ATOMS 참고
ALLOWED_ATOMS = {
    'C', 'N', 'O', 'S', 'P', 'B',
    'F', 'Cl', 'Br', 'I',
    'H', 'D', 'T'
}

def filter_organic(smiles):
    mol = Chem.MolFromSmiles(smiles)
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset(ALLOWED_ATOMS)
```

**목적**:
- 무기물 제거 (금속 이온, 희귀 원소 등)
- 독성 예측 모델의 적용 범위(applicability domain) 제한
- 데이터 품질 향상

**MolVS에는 없는 기능**: 
- MolVS는 이러한 원소 기반 필터링을 제공하지 않음
- TOXBAI 연구에서는 특정 화학 공간(organic subset)에 집중하기 위해 추가됨

---

### 4. Custom Salt List

#### TOXBAI의 Salts.txt:
```
[Cl-]
[Br-]
[Na+]
[K+]
```

**특징**:
- 매우 간단한 salt 리스트 (4개 패턴만)
- 독성 데이터에서 자주 등장하는 간단한 이온만 포함
- 필요에 따라 쉽게 확장 가능

#### MolVS의 기본 Salt List:
- 훨씬 더 많은 salt 패턴 포함 (~50개 이상)
- 약학적으로 중요한 다양한 salt 포함
- 일반적인 용도로 설계됨

**선택 기준**:
- **TOXBAI 방식**: 특정 연구/데이터셋에 맞춤화
- **MolVS 방식**: 범용적 사용

---

## 🎯 통합 코드의 장점

새로운 `standardize_smiles.py`는 두 방식의 장점을 결합:

```python
def standardize_smiles(
    smiles: str,
    use_molvs: bool = True,      # MolVS standardization
    remove_salts: bool = True,    # TOXBAI salt removal
    filter_organics: bool = True, # TOXBAI organic filter
    salt_file: str = None
) -> Optional[str]:
    """
    1. (Optional) MolVS standardization
    2. TOXBAI-style salt removal
    3. Organic filtering
    """
```

### 유연한 사용 시나리오:

#### 시나리오 1: QSAR 모델링 (최대 표준화)
```python
result = standardize_smiles(
    smiles,
    use_molvs=True,      # ✓ Tautomer 정규화
    remove_salts=True,   # ✓ Salt 제거
    filter_organics=True # ✓ Organic만
)
```

#### 시나리오 2: TOXBAI 방식 (원본 구조 보존)
```python
result = standardize_smiles_toxbai(smiles, salt_file="Salts.txt")
# = standardize_smiles(smiles, use_molvs=False, remove_salts=True, filter_organics=True)
```

#### 시나리오 3: 간단한 정리만
```python
result = standardize_smiles_simple(smiles)
# = standardize_smiles(smiles, use_molvs=True, remove_salts=False, filter_organics=False)
```

---

## 📦 의존성

### 기존 코드 (MolVS only):
```txt
rdkit
molvs
```

### TOXBAI 방식:
```txt
rdkit-pypi
pandas
tqdm
```

### 통합 코드:
```txt
rdkit-pypi  # 또는 rdkit
molvs       # (optional, use_molvs=True일 때만 필요)
```

---

## 🔬 사용 예시

```python
from standardize_smiles import standardize_smiles, compare_approaches

# 예시 1: Salt가 포함된 SMILES
smiles = "CC(=O)[O-].[Na+]"

# 모든 방식으로 비교
results = compare_approaches(smiles)
print(results)
# {
#     'original': 'CC(=O)[O-].[Na+]',
#     'molvs_only': 'CC(=O)O',           # 전하 중성화 + salt 제거
#     'toxbai_only': 'CC(=O)[O-]',       # [Na+]만 제거
#     'combined': 'CC(=O)O'              # 완전 표준화
# }

# 예시 2: 커스텀 salt list 사용
result = standardize_smiles(
    smiles,
    salt_file="custom_salts.txt",
    use_molvs=True,
    remove_salts=True,
    filter_organics=True
)

# 예시 3: 무기물 필터링 테스트
inorganic = "O=[Fe]=O"  # Iron oxide
result = standardize_smiles(inorganic, filter_organics=True)
print(result)  # None (Fe는 허용되지 않음)
```

---

## 💡 권장사항

### TOXBAI 방식을 선택해야 하는 경우:
- ✅ TOXBAI 논문/연구를 재현하는 경우
- ✅ 독성 예측 모델 학습 데이터 전처리
- ✅ 원본 구조를 최대한 보존하면서 정리만 필요한 경우
- ✅ 특정 화학 공간(organic subset)에 집중하는 경우

### MolVS 방식을 선택해야 하는 경우:
- ✅ 일반적인 QSAR 모델링
- ✅ 화학 구조 데이터베이스 중복 제거
- ✅ Tautomer 정규화가 중요한 경우
- ✅ 업계 표준 전처리가 필요한 경우

### 통합 방식 (추천):
- ✅ 최상의 표준화가 필요한 경우
- ✅ 유연한 파라미터 조정이 필요한 경우
- ✅ 두 방식의 장점을 모두 활용하고 싶은 경우

---

## 📝 결론

TOXBAI 리포지토리의 전처리는 **독성 예측에 특화된 간단하고 명확한 접근법**을 제공합니다. 
반면 MolVS는 **범용적이고 포괄적인 화학 구조 표준화**를 제공합니다.

새로운 통합 코드는 두 방식을 모두 지원하여, 사용자가 상황에 맞게 선택할 수 있도록 합니다.
