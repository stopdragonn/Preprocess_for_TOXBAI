# QSAR Preprocessing Pipeline for Toxicology (TOXBAI)

> **Comprehensive SMILES preprocessing + 400+ molecular descriptors for QSAR/toxicology modeling**
> 
> **3-Stage Pipeline**: File Loading → SMILES Preprocessing → Descriptor Calculation

---

## 📋 Features

✅ **Stage 1: File Loading**
- Supports Excel (.xlsx, .xls) and CSV formats
- Automatic test_data folder recognition
- File preview with data validation

✅ **Stage 2: SMILES Preprocessing**
- MolVS standardization (tautomer normalization)
- Salt removal (TOXBAI SMARTS patterns)
- pH 7.4 protomer normalization
- Organic molecule filtering
- Stereochemistry preservation
- Pass rate statistics & visualization

✅ **Stage 3: Molecular Descriptors (400+)**
- **192** AUTOCORR2D descriptors (2D autocorrelation)
- **60+** Functional group descriptors (drug metabolism)
- **80+** Basic descriptors (MW, LogP, TPSA, etc.)
- **40+** VSA descriptors (surface area based)
- **12** Chi connectivity indices
- **8** BCUT2D eigenvalue descriptors
- **And many more...**

---

## 🚀 Quick Start

### Installation
```bash
# 1. Clone repository
git clone https://github.com/stopdragonn/Preprocess_for_TOXBAI.git
cd Preprocess_for_TOXBAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Notebook
jupyter notebook QSAR_Preprocessing_Pipeline.ipynb
```

### Notebook Workflow (5 Steps)

**Step 1: Import Libraries** (Cell 1)
- Click "Run" to load all required packages

**Step 2: Generate Test Data** (Cell 3, Optional)
- Click button to create 10 sample molecules in `test_data/`
- Useful for testing before using your own data

**Step 3: Load File** (Cell 2)
- Enter path: `test_data/test_molecules.xlsx` (default)
- Click "Load File" button
- Review data preview

**Step 4: Run Preprocessing** (Cell 7)
- Enter SMILES column name: `SMILES` (default)
- Click "Run Preprocessing" button
- View 4-panel statistics dashboard
- Results saved to `preprocessed_data/` folder

**Step 5: Calculate Descriptors** (Cell 11)
- Click "Calculate Descriptors" button
- Waits for Step 4 completion
- Processes 400+ descriptors per molecule
- Results saved to `molecular_descriptors/` folder

### Python Script Usage
```python
import pandas as pd
from qsar_preprocess import QSARPreprocessor

# Load your data
df = pd.read_excel("molecules.xlsx")

# Create preprocessor
preprocessor = QSARPreprocessor(
    use_molvs=True,
    remove_salts=True,
    filter_organics=True
)

# Run preprocessing
df_clean = preprocessor.preprocess_dataframe(
    df, 
    smiles_column="SMILES",
    keep_original=True,
    drop_invalid=True
)

# Save results
df_clean.to_csv("preprocessed.csv", index=False)
```

---

## 📁 Project Structure

```
Preprocess_for_TOXBAI/
├── QSAR_Preprocessing_Pipeline.ipynb   ⭐ Main Jupyter notebook
├── qsar_preprocess.py                  Preprocessing module
├── standardize_smiles.py               SMILES utilities
├── requirements.txt                    Dependencies
├── Salts_extended.txt                  Salt patterns
├── README.md                           Documentation
│
├── test_data/                          📂 Input folder
│   └── test_molecules.xlsx
│
├── preprocessed_data/                  📂 Stage 1 output (auto-created)
│   └── preprocessed_*.csv
│
└── molecular_descriptors/              📂 Stage 2 output (auto-created)
    └── descriptors_*.csv
```

---

## ⚙️ Default Settings

| Setting | Value |
|---------|-------|
| **MolVS Standardization** | ✓ Enabled |
| **Salt Removal** | ✓ Enabled |
| **pH 7.4 Protomer** | ✓ Enabled |
| **Organic Filter** | ✓ Enabled |
| **Stereochemistry** | ✓ Preserved |

---

## 📊 Output Format

### Stage 1: `preprocessed_data/preprocessed_YYYYMMDD_HHMMSS.csv`
```
Name, SMILES, SMILES_clean, Activity, ...
Aspirin, CC(=O)Oc1ccccc1C(=O)O, CC(=O)Oc1ccccc1C(=O)O, 0.8, ...
```

### Stage 2: `molecular_descriptors/descriptors_YYYYMMDD_HHMMSS.csv`
```
SMILES_clean, MolWt, LogP, TPSA, AUTOCORR2D_1, ..., AUTOCORR2D_192, ...
CC(=O)Oc1ccccc1C(=O)O, 180.16, 1.19, 63.6, 1.2, ..., 0.5, ...
```

---

## 📈 Descriptor Categories (400+)

| Category | Count | Example |
|----------|-------|---------|
| **AUTOCORR2D** | 192 | 2D spatial descriptors |
| **Functional Groups** | 60+ | fr_aldehyde, fr_ketone, fr_amide |
| **Basic** | 80+ | MolWt, LogP, TPSA, NumHDonors |
| **VSA** | 40+ | PEOE_VSA1-14, EState_VSA1-11 |
| **Chi** | 12 | Chi0, Chi1, Chi1v, Chi2v, ... |
| **BCUT2D** | 8 | BCUT2D_MWLOW, BCUT2D_MWHIGH, ... |
| **Other** | 8+ | Kappa1, Kappa2, HallKierAlpha, ... |

---

## 🛠️ Dependencies

```bash
rdkit>=2022.09          # Chemistry & molecular descriptors
pandas>=1.3.0           # Data manipulation
numpy>=1.20.0           # Numerical computing
molvs>=0.1.1            # SMILES standardization
matplotlib>=3.3.0       # Visualization
seaborn>=0.11.0         # Statistical plots
jupyter>=1.0.0          # Notebook environment
ipywidgets>=8.0.0       # Interactive widgets
openpyxl>=3.0.0         # Excel file support
```

---

## 🔑 Key Features

### ✅ **Easy to Use**
- Interactive Jupyter interface with buttons
- No command-line knowledge required
- Point-and-click workflow

### ✅ **Comprehensive**
- 400+ descriptors for QSAR modeling
- 60+ functional groups for drug metabolism
- 4-panel visualization dashboard

### ✅ **Robust**
- Error handling for invalid SMILES
- Quality control with pass rate statistics
- Organized timestamped outputs

### ✅ **Flexible**
- Excel & CSV support
- Preserves original columns & stereochemistry
- Customizable preprocessing options

---

## 📥 Input Format

### Requirements
- SMILES column (default name: `SMILES`)
- Additional columns are preserved
- One molecule per row

### Example
| Name | SMILES | Activity |
|------|--------|----------|
| Aspirin | CC(=O)Oc1ccccc1C(=O)O | 0.8 |
| Caffeine | CN1C=NC2=C1C(=O)N(C(=O)N2C)C | 0.6 |

---

## 📞 Troubleshooting

**Issue**: File not found
- Solution: Place Excel/CSV in `test_data/` folder or current directory

**Issue**: "SMILES column not found"
- Solution: Verify column name matches (case-sensitive)

**Issue**: Descriptor calculation fails
- Solution: Run preprocessing (Step 4) first

**Issue**: Missing dependencies
- Solution: `pip install -r requirements.txt`

---

## 📄 License

See LICENSE file

---

**Version**: 2.0 (400+ Descriptors)  
**Updated**: 2025-12-17  
**Status**: ✅ Production Ready
