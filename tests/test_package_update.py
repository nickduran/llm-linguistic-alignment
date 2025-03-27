import sys
import os
import pandas as pd

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import LinguisticAlignment

# Define path to data folder - adjust paths as needed for your environment
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_mid")
output_folder = "tests/results"

# Initialize with one or more alignment types: Choose one or more: "bert", "fasttext", or "lexsyn"
analyzer = LinguisticAlignment(
    alignment_types=["bert", "lexsyn"],  # Run multiple analyzers
    cache_dir=os.path.join(output_folder, "cache")
)

# Analyze real conversations with all parameters in one call
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,
    lag=1,
    # FastText parameters
    high_sd_cutoff=3,
    low_n_cutoff=1,
    save_vocab=True,
    # LexSyn parameters 
    max_ngram=3,
    ignore_duplicates=True,
    add_stanford_tags=True
)

# Analyze baseline with the same parameters
baseline_results = analyzer.analyze_baseline(
    input_files=data_path,
    output_directory=output_folder,
    # File format parameters
    id_separator="_",
    condition_label="ExpBlock",
    dyad_label="ASU-",
    # Other parameters from above apply automatically
)

# # Optional: use existing surrogates
# alt_baseline = analyzer.analyze_baseline(
#     input_files=data_path,
#     output_directory=output_folder,
#     use_existing_surrogates=os.path.join(output_folder, "surrogates/surrogate_run-1234567890"),
#     # Other parameters...
# )