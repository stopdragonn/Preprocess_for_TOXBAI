"""
Example usage of the enhanced standardize_smiles module

이 스크립트는 새로운 standardize_smiles.py 모듈의 다양한 사용 방법을 보여줍니다.
"""

from standardize_smiles import (
    standardize_smiles,
    standardize_smiles_simple,
    standardize_smiles_toxbai,
    compare_approaches,
)


def example_1_basic_usage():
    """기본 사용 예시"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    test_smiles = [
        "CC(=O)[O-].[Na+]",  # Sodium acetate (salt)
        "c1ccccc1",          # Benzene
        "CCO",               # Ethanol
    ]
    
    for smiles in test_smiles:
        result = standardize_smiles(smiles)
        print(f"Input:  {smiles}")
        print(f"Output: {result}")
        print()


def example_2_comparison():
    """다양한 방식 비교"""
    print("=" * 60)
    print("Example 2: Comparing Different Approaches")
    print("=" * 60)
    
    # Salt가 포함된 복잡한 예시
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C.[Cl-].[Na+]"
    print(f"Original SMILES: {smiles}")
    print("(Caffeine with NaCl)")
    print()
    
    results = compare_approaches(smiles)
    
    print(f"MolVS only:     {results['molvs_only']}")
    print(f"TOXBAI only:    {results['toxbai_only']}")
    print(f"Combined:       {results['combined']}")
    print()


def example_3_custom_parameters():
    """파라미터 커스터마이징"""
    print("=" * 60)
    print("Example 3: Custom Parameters")
    print("=" * 60)
    
    smiles = "CC(=O)[O-].[K+]"
    print(f"Input: {smiles}")
    print()
    
    # Only MolVS standardization
    result1 = standardize_smiles(
        smiles,
        use_molvs=True,
        remove_salts=False,
        filter_organics=False
    )
    print(f"MolVS only (no salt removal): {result1}")
    
    # Only salt removal
    result2 = standardize_smiles(
        smiles,
        use_molvs=False,
        remove_salts=True,
        filter_organics=False
    )
    print(f"Salt removal only:            {result2}")
    
    # Full pipeline
    result3 = standardize_smiles(
        smiles,
        use_molvs=True,
        remove_salts=True,
        filter_organics=True
    )
    print(f"Full pipeline:                {result3}")
    print()


def example_4_organic_filtering():
    """Organic 필터링 예시"""
    print("=" * 60)
    print("Example 4: Organic Filtering")
    print("=" * 60)
    
    test_cases = [
        ("CCO", "Ethanol (organic)"),
        ("O=[Fe]=O", "Iron oxide (inorganic)"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen (organic)"),
        ("[Cu+2]", "Copper ion (inorganic)"),
    ]
    
    for smiles, name in test_cases:
        # With organic filtering
        result_filtered = standardize_smiles(
            smiles,
            use_molvs=False,
            remove_salts=False,
            filter_organics=True
        )
        
        # Without organic filtering
        result_unfiltered = standardize_smiles(
            smiles,
            use_molvs=False,
            remove_salts=False,
            filter_organics=False
        )
        
        print(f"{name}:")
        print(f"  Input:                {smiles}")
        print(f"  Without filtering:    {result_unfiltered}")
        print(f"  With organic filter:  {result_filtered}")
        print()


def example_5_batch_processing():
    """여러 SMILES를 한번에 처리"""
    print("=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    smiles_list = [
        "CC(=O)O.[Na+]",
        "c1ccccc1",
        "CCO",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
    ]
    
    results = []
    failed = []
    
    for smiles in smiles_list:
        result = standardize_smiles(smiles)
        if result:
            results.append((smiles, result))
        else:
            failed.append(smiles)
    
    print(f"Successfully processed: {len(results)}/{len(smiles_list)}")
    print(f"Failed: {len(failed)}")
    print()
    
    print("Results:")
    for original, standardized in results:
        print(f"  {original:50s} -> {standardized}")
    
    if failed:
        print("\nFailed SMILES:")
        for smiles in failed:
            print(f"  {smiles}")
    print()


def example_6_toxbai_style():
    """TOXBAI 스타일 전처리 (MolVS 없이)"""
    print("=" * 60)
    print("Example 6: TOXBAI-Style Preprocessing")
    print("=" * 60)
    
    smiles = "CC(=O)[O-].[Na+]"
    print(f"Input: {smiles}")
    print()
    
    # TOXBAI 방식 (MolVS 사용 안함)
    result = standardize_smiles_toxbai(smiles)
    print(f"TOXBAI style: {result}")
    
    # 비교: MolVS 포함
    result_with_molvs = standardize_smiles(smiles)
    print(f"With MolVS:   {result_with_molvs}")
    print()
    
    print("Note: TOXBAI 방식은 원본 구조를 더 보존합니다.")
    print("      MolVS는 추가로 전하 중성화, tautomer 정규화 등을 수행합니다.")
    print()


def example_7_error_handling():
    """에러 처리 예시"""
    print("=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)
    
    invalid_smiles = [
        "INVALID",
        "",
        None,
        "C1CCCCC",  # Incomplete ring
    ]
    
    for smiles in invalid_smiles:
        result = standardize_smiles(smiles)
        print(f"Input:  {repr(smiles):30s} -> Output: {result}")
    print()


def main():
    """모든 예시 실행"""
    examples = [
        example_1_basic_usage,
        example_2_comparison,
        example_3_custom_parameters,
        example_4_organic_filtering,
        example_5_batch_processing,
        example_6_toxbai_style,
        example_7_error_handling,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Error in example {i}: {e}")
            print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
