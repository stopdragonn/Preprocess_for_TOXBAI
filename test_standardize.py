"""
Test suite for standardize_smiles module

Tests all major functionality to ensure correctness
"""

import sys
from standardize_smiles import (
    standardize_smiles,
    standardize_smiles_simple,
    standardize_smiles_toxbai,
    compare_approaches,
    filter_organic,
    strip_salts_toxbai,
    load_salt_smarts_list,
)


def test_salt_removal():
    """Test TOXBAI-style salt removal"""
    print("Testing salt removal...")
    
    # Test with simple salt
    result = standardize_smiles("CC(=O)[O-].[Na+]", use_molvs=False, remove_salts=True, filter_organics=False)
    assert result is not None, "Salt removal should not return None for valid input"
    assert "[Na+]" not in result, "Sodium ion should be removed"
    print(f"  ✓ Simple salt removal: CC(=O)[O-].[Na+] -> {result}")
    
    # Test with multiple salts
    result = standardize_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C.[Cl-].[Na+]", 
                                use_molvs=False, remove_salts=True, filter_organics=False)
    assert result is not None, "Should handle multiple salts"
    assert "[Na+]" not in result and "[Cl-]" not in result, "All salts should be removed"
    print(f"  ✓ Multiple salt removal: Caffeine+NaCl -> {result}")
    
    print("Salt removal tests passed!\n")


def test_organic_filtering():
    """Test organic molecule filtering"""
    print("Testing organic filtering...")
    
    # Organic molecule (should pass)
    assert filter_organic("CCO"), "Ethanol should be classified as organic"
    print("  ✓ Ethanol is organic")
    
    # Inorganic molecule (should fail)
    assert not filter_organic("O=[Fe]=O"), "Iron oxide should be classified as inorganic"
    print("  ✓ Iron oxide is inorganic")
    
    # Complex organic molecule
    assert filter_organic("CC(C)Cc1ccc(C(C)C(=O)O)cc1"), "Ibuprofen should be organic"
    print("  ✓ Ibuprofen is organic")
    
    # Copper ion (should fail)
    assert not filter_organic("[Cu+2]"), "Copper ion should be inorganic"
    print("  ✓ Copper ion is inorganic")
    
    print("Organic filtering tests passed!\n")


def test_standardize_modes():
    """Test different standardization modes"""
    print("Testing different standardization modes...")
    
    test_smiles = "CC(=O)[O-].[Na+]"
    
    # Mode 1: Simple (MolVS only, if available)
    result1 = standardize_smiles_simple(test_smiles)
    assert result1 is not None, "Simple mode should work"
    print(f"  ✓ Simple mode: {result1}")
    
    # Mode 2: TOXBAI only
    result2 = standardize_smiles_toxbai(test_smiles)
    assert result2 is not None, "TOXBAI mode should work"
    assert "[Na+]" not in result2, "TOXBAI mode should remove salts"
    print(f"  ✓ TOXBAI mode: {result2}")
    
    # Mode 3: Combined
    result3 = standardize_smiles(test_smiles, use_molvs=True, remove_salts=True, filter_organics=True)
    assert result3 is not None, "Combined mode should work"
    print(f"  ✓ Combined mode: {result3}")
    
    print("Standardization mode tests passed!\n")


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("Testing error handling...")
    
    invalid_inputs = [
        None,
        "",
        "INVALID_SMILES",
        "C1CCCCC",  # Incomplete ring
    ]
    
    for invalid in invalid_inputs:
        result = standardize_smiles(invalid)
        assert result is None, f"Should return None for invalid input: {repr(invalid)}"
        print(f"  ✓ Correctly handled: {repr(invalid)}")
    
    print("Error handling tests passed!\n")


def test_batch_processing():
    """Test processing multiple SMILES"""
    print("Testing batch processing...")
    
    smiles_list = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O.[Na+]",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    
    results = []
    for smiles in smiles_list:
        result = standardize_smiles(smiles)
        results.append(result)
    
    # All should succeed
    assert all(r is not None for r in results), "All valid SMILES should be processed"
    assert len(results) == len(smiles_list), "Should process all inputs"
    
    print(f"  ✓ Successfully processed {len(results)}/{len(smiles_list)} SMILES")
    print("Batch processing tests passed!\n")


def test_comparison_function():
    """Test the comparison function"""
    print("Testing comparison function...")
    
    test_smiles = "CC(=O)[O-].[Na+]"
    results = compare_approaches(test_smiles)
    
    # Check all expected keys are present
    expected_keys = ['original', 'molvs_only', 'toxbai_only', 'combined']
    for key in expected_keys:
        assert key in results, f"Missing key in results: {key}"
    
    # Original should match input
    assert results['original'] == test_smiles, "Original should match input"
    
    print("  ✓ Comparison function works correctly")
    print(f"    Original:   {results['original']}")
    print(f"    MolVS:      {results['molvs_only']}")
    print(f"    TOXBAI:     {results['toxbai_only']}")
    print(f"    Combined:   {results['combined']}")
    print("Comparison function tests passed!\n")


def test_custom_salt_file():
    """Test custom salt file loading"""
    print("Testing custom salt file loading...")
    
    # Test loading default salts
    salt_mols = load_salt_smarts_list()
    assert len(salt_mols) > 0, "Should load default salts"
    print(f"  ✓ Loaded {len(salt_mols)} default salt patterns")
    
    # Test loading from Salts.txt
    salt_mols = load_salt_smarts_list("Salts.txt")
    assert len(salt_mols) > 0, "Should load salts from file"
    print(f"  ✓ Loaded {len(salt_mols)} salts from Salts.txt")
    
    print("Custom salt file tests passed!\n")


def test_edge_cases():
    """Test edge cases and special molecules"""
    print("Testing edge cases...")
    
    test_cases = [
        ("c1ccccc1", "Aromatic benzene"),
        ("C=C", "Simple alkene"),
        ("C#C", "Alkyne"),
        ("[H]C([H])([H])C([H])([H])[H]", "Explicit hydrogens"),
    ]
    
    for smiles, description in test_cases:
        result = standardize_smiles(smiles)
        assert result is not None, f"Should handle {description}"
        print(f"  ✓ {description}: {smiles} -> {result}")
    
    print("Edge case tests passed!\n")


def run_all_tests():
    """Run all test suites"""
    print("=" * 60)
    print("Running Standardize SMILES Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_salt_removal,
        test_organic_filtering,
        test_standardize_modes,
        test_error_handling,
        test_batch_processing,
        test_comparison_function,
        test_custom_salt_file,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test failed: {test_func.__name__}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ Test error: {test_func.__name__}")
            print(f"   Error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
