# SMILES 전처리 파이프라인

> RDKit과 pandas를 이용해 SMILES 데이터에 대해 ‘염 제거 → 유기물질 선택 → 분자설명자 계산’ 단계를 자동화하는 워크플로우입니다.

## 목차

* [소개](#소개)
* [사전 요구사항](#사전-요구사항)
* [입력 파일](#입력-파일)
* [설치 방법](#설치-방법)
* [사용법](#사용법)
* [워크플로우 단계](#워크플로우-단계)

  * [1. 염 제거 (Salt Stripping)](#1-염-제거-salt-stripping)
  * [2. 유기물질 선택 (Organic Subset Filtering)](#2-유기물질-선택-organic-subset-filtering)
  * [3. 분자설명자 생성 (Descriptor Calculation)](#3-분자설명자-생성-descriptor-calculation)
* [파이프라인 예제](#파이프라인-예제)
* [저장소 구조](#저장소-구조)
* [기여](#기여)
* [라이선스](#라이선스)

---

## 소개

이 저장소는 전처리된 SMILES 데이터를 바탕으로 RDKit을 활용해 염 제거, 유기물질 필터링, 분자설명자 계산을 수행하는 Python 스크립트를 제공합니다. 후임자나 협업 연구진이 코드 구조를 빠르게 이해하고 실행할 수 있도록 설계되었습니다.

## 사전 요구사항

* Python 3.8 이상
* Conda 환경 권장
* 주요 라이브러리:

  * rdkit-pypi
  * pandas
  * tqdm

```bash
# Conda 환경 생성 예시
donda create -n chem_preprocess python=3.8
conda activate chem_preprocess
pip install rdkit-pypi pandas tqdm
```

## 입력 파일

* **Preprocessed1\_Uniq\&NaNhandling.csv**: 중복값 제거 및 결측치 처리 완료 후 SMILES 컬럼이 담긴 파일

## 설치 방법

```bash
git clone <GitHub_repo_URL>
cd <repo_folder>
conda activate chem_preprocess
pip install -r requirements.txt
```

## 사용법

```bash
python preprocess.py \
  --input Preprocessed1_Uniq&NaNhandling.csv \
  --salts Salts.txt \
  --output_dir ./outputs
```

필요에 따라 `--input` 및 `--output_dir` 경로를 조정하세요.

## 워크플로우 단계

### 1. 염 제거 (Salt Stripping)

```python
from rdkit import Chem
from rdkit.Chem import SaltRemover

remover = SaltRemover.SaltRemover(defnFile='Salts.txt', onlyUncharged=False)

def strip_salts(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    stripped = remover.StripMol(mol)
    return Chem.MolToSmiles(stripped)
```

* **입력**: `Preprocessed1_Uniq&NaNhandling.csv`
* **출력**: `Preprocessed2_Saltstripped.csv`
* **검증**: 제거 전·후 row 수 비교, 대표 SMILES 확인

### 2. 유기물질 선택 (Organic Subset Filtering)

```python
from rdkit import Chem

allowed = set(['C','N','O','S','P','B','F','Cl','Br','I','H','D','T'])

def filter_organic(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return symbols.issubset(allowed)
```

* **입력**: `Preprocessed2_Saltstripped.csv`
* **출력**: `Preprocessed3_Organicselected.csv`
* **검증**: 필터링 건수 로그 출력

### 3. 분자설명자 생성 (Descriptor Calculation)

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

descriptor_funcs = {
    'SlogP': Descriptors.MolLogP,
    'SMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
    # ...
}

def calc_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {name: None for name in descriptor_funcs}
    values = {}
    for name, func in descriptor_funcs.items():
        try:
            values[name] = func(mol)
        except:
            values[name] = None
    return values
```

* **입력**: `Preprocessed3_Organicselected.csv`
* **출력**: `Preprocessed4_DescriptorGen.csv`
* **예외처리**: 계산 실패 시 None

## 파이프라인 예제

```python
import pandas as pd
from tqdm import tqdm
from workflow import strip_salts, filter_organic, calc_descriptors

# 1) Load
df = pd.read_csv('Preprocessed1_Uniq&NaNhandling.csv')

# 2) Salt stripping
 df['smiles_stripped'] = df['SMILES'].apply(strip_salts)
 df.dropna(subset=['smiles_stripped']).to_csv('Preprocessed2_Saltstripped.csv', index=False)

# 3) Organic filtering
 filtered = df[df['smiles_stripped'].apply(filter_organic)]
 filtered.to_csv('Preprocessed3_Organicselected.csv', index=False)

# 4) Descriptor generation
 desc_list = [calc_descriptors(smi) for smi in tqdm(filtered['smiles_stripped'])]
 pd.concat([filtered.reset_index(drop=True), pd.DataFrame(desc_list)], axis=1)\
   .to_csv('Preprocessed4_DescriptorGen.csv', index=False)
```

## 저장소 구조

```
├── Salts.txt
├── preprocess.py
├── workflow.py      # 함수 정의 모듈
├── requirements.txt
├── README.md
└── outputs/
    ├── Preprocessed2_Saltstripped.csv
    ├── Preprocessed3_Organicselected.csv
    └── Preprocessed4_DescriptorGen.csv
```

## 기여

1. Fork the repository
2. 새로운 브랜치 생성
3. 변경 후 `git commit`
4. `git push` 및 Pull Request 생성

## 라이선스

MIT License
