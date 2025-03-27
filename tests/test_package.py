import sys
import os
import pandas as pd

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import LinguisticAlignment
from my_package.surrogates import SurrogateGenerator, SurrogateAlignment

# Define path to data folder - adjust paths as needed for your environment
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_mid")
output_folder = "tests/results"

# Make sure root output directory exists
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "surrogates"), exist_ok=True)

# File format parameters for your specific files
id_separator = "_"
condition_label = "ExpBlock"
dyad_label = "ASU-"

# Initialize analyzer with the desired alignment type
# Choose one: "bert", "fasttext", or "lexsyn"
alignment_type = "bert"  # Changed from "word2vec" to "fasttext"

# Create cache directory for models that need it
if alignment_type in ["fasttext", "bert"]:
    cache_dir = os.path.join(output_folder, alignment_type, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    analyzer = LinguisticAlignment(alignment_type=alignment_type, cache_dir=cache_dir)
else:
    analyzer = LinguisticAlignment(alignment_type=alignment_type)

# Configure parameters for all types of analyzers
# These will be used as needed based on the analyzer types
fasttext_params = {  # Renamed from w2v_params to fasttext_params
    "high_sd_cutoff": 3,    # Filter out words with frequency > mean + 3*std
    "low_n_cutoff": 1,      # Filter out words occurring < 1 times
    "save_vocab": True      # Save vocabulary lists to output directory
}

lexsyn_params = {
    "max_ngram": 3,
    "ignore_duplicates": True,
    "add_stanford_tags": True
}

# Parameters for any analyzer
common_params = {
    "lag": 1
}

# 1. First, analyze the real conversations
print(f"Analyzing real conversations with {alignment_type} model...")
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,  # Now uses root directory, analyzer will create model-specific subdir
    **common_params,
    **fasttext_params,  # Renamed from w2v_params
    **lexsyn_params
)

# 2. Next, generate surrogate pairs and analyze them
print(f"Generating and analyzing surrogate pairs with {alignment_type} model...")
baseline_results = analyzer.analyze_baseline(
    input_files=data_path,
    output_directory=output_folder,  # Now uses root directory, analyzer will create model-specific subdir
    surrogate_directory=os.path.join(output_folder, "surrogates"),
    all_surrogates=False,
    keep_original_turn_order=True,
    # Adjust these to match your actual filename format
    id_separator="_",              # The separator between components
    condition_label="ExpBlock",    # The string that identifies conditions
    dyad_label="ASU-",             # The string that identifies dyads
    # Include all other parameters
    **common_params,
    **fasttext_params,  # Renamed from w2v_params
    **lexsyn_params
)