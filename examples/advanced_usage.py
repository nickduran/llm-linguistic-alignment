"""
Advanced Usage Example for ALIGN Package

This script demonstrates comprehensive usage of the ALIGN package
including multiple analysis types and baseline comparison.
"""

import os
import sys

# Add src directory to path to allow importing from align_test
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from align_test.alignment import LinguisticAlignment

# Define paths for input data and output
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "src", "align_test", "data", "prepped_stan_mid")
output_folder = os.path.join(current_dir, "output_advanced")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("ALIGN Advanced Usage Example")
print("===========================")
print(f"Data path: {data_path}")
print(f"Output folder: {output_folder}")
print("----------------------------")

# Initialize with multiple alignment types
analyzer = LinguisticAlignment(
    alignment_types=["fasttext", "bert", "lexsyn"],
    cache_dir=os.path.join(output_folder, "cache")
)

# Configure parameters for FastText
fasttext_params = {
    "high_sd_cutoff": 3,    # Filter out words with frequency > mean + 3*std
    "low_n_cutoff": 2,      # Filter out words occurring < 2 times
    "save_vocab": True      # Save vocabulary lists to output directory
}

# Configure parameters for Lexical/Syntactic analysis
lexsyn_params = {
    "max_ngram": 3,         # Maximum n-gram size
    "ignore_duplicates": True,
    "add_stanford_tags": True  # Include Stanford POS tags if available
}

# Common parameters for all analyzers
common_params = {
    "lag": 1  # Number of turns to lag
}

print("\nStep 1: Analyzing real conversations")
# Analyze real conversations
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params
)

print(f"Real conversation analysis complete. Rows: {len(real_results) if real_results is not None else 0}")

# Parameters for surrogate generation
surrogate_params = {
    "all_surrogates": False,  # Generate a subset rather than all possible pairs
    "keep_original_turn_order": True,
    "id_separator": "_",
    "condition_label": "ExpBlock",  # Part of filename identifying experimental condition
    "dyad_label": "ASU-"  # Part of filename identifying conversation pair
}

print("\nStep 2: Generating and analyzing surrogate conversations")
# Analyze baseline (chance) alignment with surrogates
baseline_results = analyzer.analyze_baseline(
    input_files=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params,
    **surrogate_params
)

print(f"Baseline analysis complete. Rows: {len(baseline_results) if baseline_results is not None else 0}")

print("\nResults Summary:")
print("----------------")
print(f"Analysis complete! Check the following files in {output_folder}:")

import glob
result_files = glob.glob(os.path.join(output_folder, "*.csv"))
for file in result_files:
    print(f"- {os.path.basename(file)}")

print("\nTo compare real vs. baseline alignment:")
print("1. Look at the merged_alignment_results_*.csv file for real conversations")
print("2. Compare with baseline_alignment_*.csv files for surrogate conversations")
print("3. The difference indicates alignment above chance levels")