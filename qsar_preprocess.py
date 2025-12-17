"""
QSAR Preprocessing Pipeline
============================

Unified preprocessing pipeline for QSAR/toxicology modeling.

Combines MolVS standardization, salt removal, and TOXBAI-style filtering
into a single, easy-to-use interface for expert users.

Usage:
    >>> import pandas as pd
    >>> from qsar_preprocess import preprocess_dataframe
    >>> 
    >>> df = pd.read_csv("molecules.csv")
    >>> df_clean = preprocess_dataframe(df, smiles_column="SMILES")
    >>> df_clean.to_csv("preprocessed.csv", index=False)

Author: Computational Toxicology Lab
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from rdkit import Chem
from rdkit import RDLogger

from standardize_smiles import (
    standardize_smiles,
    normalize_protomers_ph74,
    filter_organic,
)

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


class QSARPreprocessor:
    """
    QSAR Preprocessing Pipeline
    
    Default configuration optimized for QSAR/toxicology modeling:
    - MolVS standardization (tautomer normalization)
    - TOXBAI-style salt removal (SMARTS-based)
    - pH 7.4 protomer normalization
    - Organic molecule filtering
    - Explicit stereochemistry retention
    """
    
    def __init__(
        self,
        salt_file: str = "Salts_extended.txt",
        use_molvs: bool = True,
        remove_salts: bool = True,
        filter_organics: bool = True,
        verbose: bool = True
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            salt_file: Path to salt SMARTS file (default: Salts_extended.txt)
            use_molvs: Apply MolVS standardization (default: True for QSAR)
            remove_salts: Remove counter-ions and salts (default: True)
            filter_organics: Filter to organic molecules only (default: True)
            verbose: Print processing statistics (default: True)
        """
        self.salt_file = salt_file
        self.use_molvs = use_molvs
        self.remove_salts = remove_salts
        self.filter_organics = filter_organics
        self.verbose = verbose
        
        # Statistics tracking
        self.stats = {
            'input_count': 0,
            'valid_smiles': 0,
            'after_standardization': 0,
            'after_salt_removal': 0,
            'after_protomer_norm': 0,
            'after_organic_filter': 0,
            'output_count': 0,
            'invalid_smiles': 0,
            'removed_by_salt': 0,
            'removed_by_organic': 0,
            'failed_processing': 0,
        }
    
    def preprocess_smiles(self, smiles: str) -> Optional[str]:
        """
        Preprocess single SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Preprocessed SMILES or None if invalid/filtered
        """
        if not smiles or not isinstance(smiles, str):
            self.stats['invalid_smiles'] += 1
            return None
        
        try:
            # Validate input SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.stats['invalid_smiles'] += 1
                return None
            
            self.stats['valid_smiles'] += 1
            
            # Apply standardization pipeline
            result = standardize_smiles(
                smiles,
                use_molvs=self.use_molvs,
                remove_salts=self.remove_salts,
                filter_organics=self.filter_organics,
                salt_file=self.salt_file
            )
            
            if result is None:
                self.stats['failed_processing'] += 1
                return None
            
            self.stats['output_count'] += 1
            return result
        
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Processing error for {smiles}: {e}")
            self.stats['failed_processing'] += 1
            return None
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        smiles_column: str = "SMILES",
        keep_original: bool = True,
        drop_invalid: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess DataFrame with SMILES column.
        
        QSAR 모델링을 위한 기본 워크플로우:
        1. SMILES 검증
        2. MolVS 표준화 (tautomer 정규화)
        3. 염 제거 (TOXBAI SMARTS)
        4. pH 7.4 프로토머 정규화
        5. 유기물 필터링
        6. 입체화학 보존
        
        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column (default: "SMILES")
            keep_original: Keep original SMILES column (default: True)
            drop_invalid: Drop rows with invalid output (default: True)
            
        Returns:
            Preprocessed DataFrame with "SMILES_clean" column
            
        Example:
            >>> df = pd.read_csv("molecules.csv")
            >>> preprocessor = QSARPreprocessor()
            >>> df_clean = preprocessor.preprocess_dataframe(df)
            >>> print(df_clean[['SMILES', 'SMILES_clean']])
        """
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in DataFrame")
        
        self.stats['input_count'] = len(df)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QSAR Preprocessing Pipeline")
            print(f"{'='*60}")
            print(f"Input molecules: {len(df)}")
            print(f"Configuration:")
            print(f"  - MolVS standardization: {self.use_molvs}")
            print(f"  - Salt removal: {self.remove_salts}")
            print(f"  - Organic filtering: {self.filter_organics}")
            print()
        
        # Apply preprocessing
        df_result = df.copy()
        df_result['SMILES_clean'] = df_result[smiles_column].apply(
            self.preprocess_smiles
        )
        
        # Track statistics
        self.stats['after_organic_filter'] = df_result['SMILES_clean'].notna().sum()
        
        if self.verbose:
            self._print_statistics(len(df))
        
        # Keep original SMILES if requested
        if not keep_original:
            df_result = df_result.drop(columns=[smiles_column])
        
        # Drop invalid rows if requested
        if drop_invalid:
            df_result = df_result[df_result['SMILES_clean'].notna()].reset_index(drop=True)
        
        return df_result
    
    def _print_statistics(self, total_input: int) -> None:
        """Print preprocessing statistics."""
        print(f"Results:")
        print(f"  Input molecules: {self.stats['input_count']}")
        print(f"  Valid SMILES: {self.stats['valid_smiles']}")
        print(f"  Successfully processed: {self.stats['output_count']}")
        print(f"  Pass rate: {100 * self.stats['output_count'] / max(1, total_input):.1f}%")
        print()
        print(f"Removed:")
        print(f"  Invalid SMILES: {self.stats['invalid_smiles']}")
        print(f"  Processing errors: {self.stats['failed_processing']}")
        print(f"{'='*60}\n")


def preprocess_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "SMILES",
    keep_original: bool = True,
    drop_invalid: bool = True,
    salt_file: str = "Salts_extended.txt",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Preprocess DataFrame with SMILES - Convenience Function
    
    Quick preprocessing with default QSAR configuration.
    
    Args:
        df: Input DataFrame
        smiles_column: Name of SMILES column (default: "SMILES")
        keep_original: Keep original SMILES column (default: True)
        drop_invalid: Drop rows with invalid output (default: True)
        salt_file: Path to salt SMARTS file (default: Salts_extended.txt)
        verbose: Print statistics (default: True)
        
    Returns:
        Preprocessed DataFrame with "SMILES_clean" column
        
    Example:
        >>> import pandas as pd
        >>> from qsar_preprocess import preprocess_dataframe
        >>> 
        >>> df = pd.read_csv("molecules.csv")
        >>> df_clean = preprocess_dataframe(df, smiles_column="SMILES")
        >>> df_clean.to_csv("preprocessed.csv", index=False)
    """
    preprocessor = QSARPreprocessor(
        salt_file=salt_file,
        use_molvs=True,  # Always use for QSAR
        remove_salts=True,
        filter_organics=True,
        verbose=verbose
    )
    return preprocessor.preprocess_dataframe(
        df,
        smiles_column=smiles_column,
        keep_original=keep_original,
        drop_invalid=drop_invalid
    )


def preprocess_smiles_batch(
    smiles_list: list,
    verbose: bool = False
) -> list:
    """
    Preprocess list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        verbose: Print progress (default: False)
        
    Returns:
        List of preprocessed SMILES (None for invalid)
        
    Example:
        >>> smiles = ["CC(=O)[O-].[Na+]", "c1ccccc1"]
        >>> cleaned = preprocess_smiles_batch(smiles)
        >>> print(cleaned)  # ["CC(=O)O", "c1ccccc1"]
    """
    preprocessor = QSARPreprocessor(verbose=verbose)
    return [preprocessor.preprocess_smiles(s) for s in smiles_list]


if __name__ == "__main__":
    # Example usage
    print("QSAR Preprocessing Pipeline - Example\n")
    
    # Example 1: Batch processing
    print("Example 1: Batch Processing")
    print("-" * 60)
    test_smiles = [
        "CC(=O)[O-].[Na+]",  # Sodium acetate
        "c1ccccc1",          # Benzene
        "CCO",               # Ethanol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C.[Cl-]",  # Caffeine + chloride
        "INVALID",           # Invalid SMILES
    ]
    
    results = preprocess_smiles_batch(test_smiles, verbose=False)
    for original, cleaned in zip(test_smiles, results):
        status = "[OK]" if cleaned else "[FAIL]"
        print(f"  {status}: {original:45s} -> {cleaned}")
    
    print()
    
    # Example 2: DataFrame processing (if pandas available)
    print("Example 2: DataFrame Processing")
    print("-" * 60)
    df = pd.DataFrame({
        'SMILES': test_smiles,
        'Name': ['Acetate', 'Benzene', 'Ethanol', 'Caffeine+Cl', 'Invalid']
    })
    
    df_clean = preprocess_dataframe(df, smiles_column='SMILES', verbose=True)
    print("\nPreprocessed DataFrame:")
    print(df_clean[['Name', 'SMILES', 'SMILES_clean']])
