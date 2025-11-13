"""
Test Script for Refactored prepare_transcripts.py

This script tests the refactored preprocessing module with the sample data files
(time200-cond1.txt and time210-cond1.txt) and verifies compatibility with
alignment analysis scripts.
"""

import os
import sys
import pandas as pd
import ast

# Add src directory to path (adjust as needed for your setup)
sys.path.append("path/to/src")

from prepare_transcripts import prepare_transcripts
from align_test.alignment import LinguisticAlignment

def test_output_format(filepath):
    """
    Verify that output file format is compatible with alignment scripts.
    """
    print(f"\n{'='*60}")
    print(f"Testing Output Format: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # Load the file
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    
    # Test 1: Check required columns exist
    print("\nTest 1: Checking required columns...")
    required_cols = ['participant', 'content', 'token', 'lemma', 'tagged_token', 'tagged_lemma', 'file']
    for col in required_cols:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} - MISSING!")
            return False
    
    # Test 2: Check data types (should all be strings for list columns)
    print("\nTest 2: Checking data types...")
    list_columns = ['token', 'lemma', 'tagged_token', 'tagged_lemma']
    for col in list_columns:
        if col in df.columns:
            first_val = df[col].iloc[0]
            if isinstance(first_val, str):
                print(f"  ✓ {col} is string: {first_val[:50]}...")
            else:
                print(f"  ✗ {col} is {type(first_val)} (should be string)!")
                return False
    
    # Test 3: Check ast.literal_eval compatibility
    print("\nTest 3: Checking ast.literal_eval compatibility...")
    for col in list_columns:
        if col in df.columns:
            try:
                parsed = ast.literal_eval(df[col].iloc[0])
                print(f"  ✓ {col} parses to {type(parsed)}: {parsed[:3] if len(parsed) > 3 else parsed}")
            except Exception as e:
                print(f"  ✗ {col} parse failed: {e}")
                return False
    
    # Test 4: Check tagged columns have correct format
    print("\nTest 4: Checking POS tag format...")
    for col in ['tagged_token', 'tagged_lemma']:
        if col in df.columns:
            try:
                parsed = ast.literal_eval(df[col].iloc[0])
                if parsed and isinstance(parsed[0], tuple) and len(parsed[0]) == 2:
                    print(f"  ✓ {col} has correct format: {parsed[0]}")
                else:
                    print(f"  ✗ {col} has incorrect format")
                    return False
            except Exception as e:
                print(f"  ✗ {col} format check failed: {e}")
                return False
    
    print("\n✓ All format tests passed!")
    return True


def test_preprocessing_basic():
    """
    Test basic preprocessing (NLTK only).
    """
    print("\n" + "="*60)
    print("TEST 1: Basic Preprocessing (NLTK only)")
    print("="*60)
    
    # Create test output directory
    output_dir = "./test_output_basic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run preprocessing
    results = prepare_transcripts(
        input_files="./",  # Current directory with sample files
        output_file_directory=output_dir,
        run_spell_check=False,  # Disable for faster testing
        minwords=2,
        add_stanford_tags=False,  # NLTK only
        input_as_directory=True
    )
    
    print(f"\nProcessed {len(results)} utterances")
    print(f"Columns: {results.columns.tolist()}")
    
    # Test output format
    test_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    if test_files:
        test_file = os.path.join(output_dir, test_files[0])
        success = test_output_format(test_file)
        return success
    else:
        print("No output files generated!")
        return False


def test_preprocessing_spacy():
    """
    Test preprocessing with spaCy tagging.
    """
    print("\n" + "="*60)
    print("TEST 2: Preprocessing with spaCy")
    print("="*60)
    
    # Create test output directory
    output_dir = "./test_output_spacy"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run preprocessing
        results = prepare_transcripts(
            input_files="./",
            output_file_directory=output_dir,
            run_spell_check=False,
            minwords=2,
            add_stanford_tags=True,
            stanford_tagger_type='spacy',  # Use spaCy
            input_as_directory=True
        )
        
        print(f"\nProcessed {len(results)} utterances")
        print(f"Columns: {results.columns.tolist()}")
        
        # Verify tagged_stan_* columns exist
        if 'tagged_stan_token' in results.columns and 'tagged_stan_lemma' in results.columns:
            print("\n✓ spaCy tagging columns present")
        else:
            print("\n✗ spaCy tagging columns missing!")
            return False
        
        # Test output format
        test_files = [f for f in os.listdir(output_dir) if f.endswith('.txt') and f != 'align_concatenated_dataframe.txt']
        if test_files:
            test_file = os.path.join(output_dir, test_files[0])
            success = test_output_format(test_file)
            return success
        else:
            print("No output files generated!")
            return False
            
    except Exception as e:
        print(f"\n✗ spaCy test failed: {e}")
        print("Note: spaCy may not be installed. Install with: pip install spacy")
        print("Then run: python -m spacy download en_core_web_sm")
        return False


def test_alignment_integration():
    """
    Test integration with alignment analysis scripts.
    """
    print("\n" + "="*60)
    print("TEST 3: Integration with Alignment Analysis")
    print("="*60)
    
    # Use output from basic test
    preprocessed_dir = "./test_output_basic"
    results_dir = "./test_alignment_results"
    
    if not os.path.exists(preprocessed_dir):
        print("No preprocessed data found. Run basic test first.")
        return False
    
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Initialize alignment analyzer
        print("\nInitializing alignment analyzer...")
        analyzer = LinguisticAlignment(
            alignment_type="lexsyn",
            cache_dir=os.path.join(results_dir, "cache")
        )
        
        # Run alignment analysis
        print("Running alignment analysis...")
        results = analyzer.analyze_folder(
            folder_path=preprocessed_dir,
            output_directory=results_dir,
            lag=1,
            max_ngram=2,
            ignore_duplicates=True,
            add_stanford_tags=False
        )
        
        if results.empty:
            print("\n✗ Alignment analysis returned no results")
            return False
        
        print(f"\n✓ Alignment analysis successful!")
        print(f"  - Analyzed {len(results)} utterance pairs")
        print(f"  - Output saved to: {results_dir}")
        
        # Check for expected columns
        expected_metrics = ['lexical_tok1_cosine', 'lexical_lem1_cosine', 
                          'pos_tok1_cosine', 'pos_lem1_cosine']
        found_metrics = [col for col in expected_metrics if col in results.columns]
        
        print(f"  - Found {len(found_metrics)}/{len(expected_metrics)} expected metrics")
        
        if len(found_metrics) == len(expected_metrics):
            print("\n✓ Integration test passed!")
            return True
        else:
            print("\n✗ Some expected metrics missing")
            return False
            
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """
    Run all tests and report results.
    """
    print("\n" + "="*60)
    print("PREPARE_TRANSCRIPTS REFACTORED - TEST SUITE")
    print("="*60)
    
    # Check if sample files exist
    sample_files = ['time200-cond1.txt', 'time210-cond1.txt']
    missing_files = [f for f in sample_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\nError: Sample files not found: {missing_files}")
        print("Please ensure sample files are in the current directory.")
        return
    
    print(f"\nFound sample files: {sample_files}")
    
    # Run tests
    results = {
        "Basic Preprocessing (NLTK)": test_preprocessing_basic(),
        "spaCy Integration": test_preprocessing_spacy(),
        "Alignment Integration": test_alignment_integration()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The refactored code is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
