"""
Comprehensive Example Script for ALIGN Package

This script demonstrates how to use the ALIGN package for comprehensive 
alignment analysis, including multiple analyzer types and baseline comparison.
"""

import os
import sys

# Add src directory to path to allow importing from align_test
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from align_test.alignment import LinguisticAlignment

# Define path to data folder 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "src", "align_test", "data", "prepped_stan_small")
output_folder = os.path.join(current_dir, "results")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize with one or more alignment types ("bert", "fasttext", or "lexsyn")
analyzer = LinguisticAlignment(
    alignment_types=["fasttext", "bert", "lexsyn"],  # Run one or multiple analyzers
    # alignment_types=["lexsyn"],  # Run one or multiple analyzers
    cache_dir=os.path.join(output_folder, "cache")
)

# Configure parameters for all types of analyzers
fasttext_params = {
    "high_sd_cutoff": 3,    # Filter out words with frequency > mean + 3*std
    "low_n_cutoff": 2,      # Filter out words occurring < 1 times
    "save_vocab": True      # Save vocabulary lists to output directory
}

lexsyn_params = {
    "max_ngram": 3,
    "ignore_duplicates": True,
    "add_stanford_tags": True
}

# Common parameters for any analyzer
common_params = {
    "lag": 3
}

# Surrogate generation parameters
surrogate_params = {
    "all_surrogates": False, 
    "keep_original_turn_order": True,
    "id_separator": "_",
    "condition_label": "ExpBlock",
    "dyad_label": "ASU-"
}

print("Step 1: Analyzing real conversations")
# Analyze real conversations
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params
)

print("Step 2: Analyzing baseline with surrogates")
# Analyze baseline
baseline_results = analyzer.analyze_baseline(
    input_files=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params,
    **surrogate_params  # Add surrogate-specific parameters
)

print("Analysis complete! Results saved to:", output_folder)

# Uncomment to use existing surrogates
# alt_baseline = analyzer.analyze_baseline(
#     input_files=data_path,
#     output_directory=output_folder,
#     use_existing_surrogates=os.path.join(output_folder, "surrogates/[directory_name]"),
#     **common_params,
#     **fasttext_params,
#     **lexsyn_params,
#     **surrogate_params
# )