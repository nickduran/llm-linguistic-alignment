import sys
import os
import pandas as pd

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import LinguisticAlignment

# Define path to data folder 
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_small")
output_folder = "tests/results2"

# Initialize with one or more alignment types ("bert", "fasttext", or "lexsyn")
analyzer = LinguisticAlignment(
    # alignment_types=["fasttext", "bert", "lexsyn"],  # Run one or multiple analyzers
    alignment_types=["lexsyn", "bert"],  # Run one or multiple analyzers
    cache_dir=os.path.join(output_folder, "cache")
)

# Configure parameters for all types of analyzers
fasttext_params = {
    "high_sd_cutoff": 3,    
    "low_n_cutoff": 2,      
    "save_vocab": True      
}

lexsyn_params = {
    "max_ngram": 3,
    "ignore_duplicates": True,
    "add_stanford_tags": False
}

# Common parameters for any analyzer
common_params = {
    "lag": 2
}

# Surrogate generation parameters
surrogate_params = {
    "all_surrogates": True, 
    "keep_original_turn_order": True,
    "id_separator": "_",
    "condition_label": "ExpBlock",
    "dyad_label": "ASU-"
}

# Analyze real conversations
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params
)

# # Analyze baseline
# baseline_results = analyzer.analyze_baseline(
#     input_files=data_path,
#     output_directory=output_folder,
#     **common_params,
#     **fasttext_params,
#     **lexsyn_params,
#     **surrogate_params  # Add surrogate-specific parameters
# )

# # Optional: use existing surrogates
# alt_baseline = analyzer.analyze_baseline(
#     input_files=data_path,
#     output_directory=output_folder,
#     use_existing_surrogates=os.path.join(output_folder, "surrogates/surrogate_run-1743180264.262242"),
#     **common_params,
#     **fasttext_params,
#     **lexsyn_params,
#     **surrogate_params
# )


