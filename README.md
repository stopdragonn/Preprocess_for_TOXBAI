# SMILES 전처리 파이프라인

> RDKit과 pandas를 이용해 SMILES 데이터에 대해 ‘염 제거 → 유기물질 선택 → (선택적) 분자설명자 계산’ 단계를 자동화

## 목차

* [소개](#소개)
* [사전 요구사항](#사전-요구사항)
* [입력 파일](#입력-파일)
* [설치 방법](#설치-방법)
* [사용법](#사용법)
* [워크플로우 단계](#워크플로우-단계)

  * [1. 염 제거 (Salt Stripping)](#1-염-제거-salt-stripping)
  * [2. 유기물질 선택 (Organic Subset Filtering)](#2-유기물질-선택-organic-subset-filtering)
  * [3. (선택적) 분자설명자 생성 (Descriptor Calculation)](#3-선택적-분자설명자-생성-descriptor-calculation)
* [파이프라인 예제](#파이프라인-예제)
* [저장소 구조](#저장소-구조)
* [기여](#기여)
* [라이선스](#라이선스)

---

## 소개

전처리된 SMILES 데이터를 바탕으로 RDKit을 활용해 염 제거, 유기물질 필터링 단계를 수행하고, 필요 시 분자설명자 계산 단계를 추가로 실행합니다.

## 사전 요구사항

* Python 3.8 이상
* Conda 환경 권장
* 주요 라이브러리:

  * rdkit-pypi
  * pandas
  * tqdm

```bash
# Conda 환경 생성 예시
conda create -n chem_preprocess python=3.8
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

기본 전처리만 실행:

```bash
python preprocess.py \
  --input Preprocessed1_Uniq&NaNhandling.csv \
  --output_dir ./outputs \
  --smiles-col SMILES \
  --n-procs 4 \
  --chunksize 100
```

분자설명자 계산까지 포함 실행:

```bash
python preprocess.py \
  --input Preprocessed1_Uniq&NaNhandling.csv \
  --output_dir ./outputs \
  --compute-descriptors \
  --smiles-col SMILES \
  --n-procs 4 \
  --chunksize 100
```

`--salts` 옵션은 지정하지 않으면 `Salts.txt`를 사용합니다. 필요에 따라 `--input`, `--output_dir`, `--compute-descriptors`, `--smiles-col` 옵션을 조정하세요.
병렬 처리를 위해 `--n-procs` 값(기본 1)을 늘리고, `--chunksize` 값(기본 100)을 조정해 성능을 최적화할 수 있습니다.

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
    
    # 분자 구조 유효성 검증 추가
    try:
        Chem.SanitizeMol(stripped)
        result_smiles = Chem.MolToSmiles(stripped)
        validation_mol = Chem.MolFromSmiles(result_smiles)
        if validation_mol is None:
            return None
        return result_smiles
    except:
        return None
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

### 3. (선택적) 분자설명자 생성 (Descriptor Calculation)

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# 확장된 descriptor 리스트 (총 119개)
descriptor_funcs = {
    'SlogP': Descriptors.MolLogP,
    'SMR': Descriptors.MolMR,
    'LabuteASA': Descriptors.LabuteASA,
    'TPSA': Descriptors.TPSA,
    'AMW': Descriptors.MolWt,
    'ExactMW': Descriptors.ExactMolWt,
    'NumLipinskiHBA': Descriptors.NumHAcceptors,
    'NumLipinskiHBD': Descriptors.NumHDonors,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    # ... VSA descriptors (slogp_VSA1-12, smr_VSA1-10, peoe_VSA1-14)
    # ... MQN descriptors (MQN1-42)
    # ... 총 119개 descriptor 지원
}

def calc_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {name: None for name in descriptor_funcs}
    values = {}
    
    # 기본 descriptor 계산
    for name, func in descriptor_funcs.items():
        try:
            values[name] = func(mol)
        except:
            values[name] = None
    
    # MQN descriptor 계산 (42개)
    try:
        mqns = rdMolDescriptors.MQNs_(mol)
        for i in range(42):
            values[f'MQN{i+1}'] = mqns[i] if i < len(mqns) else None
    except:
        for i in range(42):
            values[f'MQN{i+1}'] = None
    
    return values
```

* **입력**: `Preprocessed3_Organicselected.csv`
* **출력**: `Preprocessed4_DescriptorGen.csv` (해당 옵션 실행 시)
* **예외처리**: 계산 실패 시 None

## 파이프라인 예제

```python
import pandas as pd
from workflow import (
    load_salt_remover,
    strip_salts,
    filter_organic,
    calc_descriptors,
    parallel_series_apply,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--salts', default='Salts.txt')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--compute-descriptors', action='store_true')
parser.add_argument('--smiles-col', default='SMILES')
parser.add_argument('--n-procs', type=int, default=1)
parser.add_argument('--chunksize', type=int, default=100)
args = parser.parse_args()

# 1) Load
df = pd.read_csv(args.input)

# 2) Salt stripping
remover = load_salt_remover(args.salts)
df['smiles_stripped'] = parallel_series_apply(
    df[args.smiles_col],
    lambda s: strip_salts(s, remover),
    n_procs=args.n_procs,
    desc='salt_strip',
    chunksize=args.chunksize,
)
filtered1 = df.dropna(subset=['smiles_stripped'])
filtered1.to_csv(
    f"{args.output_dir}/Preprocessed2_Saltstripped.csv",
    index=False,
)

# 3) Organic filtering
 mask = parallel_series_apply(
     filtered1['smiles_stripped'],
     filter_organic,
     n_procs=args.n_procs,
     desc='organic_filter',
     chunksize=args.chunksize,
 )
filtered2 = filtered1[mask]
filtered2.to_csv(
    f"{args.output_dir}/Preprocessed3_Organicselected.csv",
    index=False,
)

# 4) Optional: Descriptor generation
 if args.compute_descriptors:
     desc_df = parallel_series_apply(
         filtered2['smiles_stripped'],
         calc_descriptors,
         n_procs=args.n_procs,
         desc='descriptors',
         chunksize=args.chunksize,
     ).apply(pd.Series)
     pd.concat([filtered2.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)\
       .to_csv(f"{args.output_dir}/Preprocessed4_DescriptorGen.csv", index=False)
```

## 저장소 구조

```
├── Salts.txt
├── preprocess.py  # 파이프라인 실행 스크립트
├── workflow.py     # 함수 정의 모듈
├── requirements.txt
├── README.md
└── outputs/
    ├── Preprocessed2_Saltstripped.csv
    ├── Preprocessed3_Organicselected.csv
    └── Preprocessed4_DescriptorGen.csv  # 옵션 실행 시
```

## 기여

1. Fork the repository
2. 새로운 브랜치 생성
3. 변경 후 `git commit`
4. `git push` 및 Pull Request 생성

## 라이선스

MIT License
