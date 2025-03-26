import sys
import os
import pandas as pd

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import SemanticAlignment
from my_package.surrogates import SurrogateGenerator, SurrogateAlignment

# Define path to data folder - adjust paths as needed for your environment
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan")
output_folder = "tests/results"

# Make sure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "real"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "baseline"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "surrogates"), exist_ok=True)

# File format parameters for your specific files
id_separator = "_"
condition_label = "ExpBlock"
dyad_label = "ASU-"

# Initialize analyzer with the desired embedding model
# Choose one: "bert", "word2vec", or "lexsyn"
embedding_model = "lexsyn"  # Change this as needed
analyzer = SemanticAlignment(embedding_model=embedding_model)

# Configure parameters for all types of embedding models
# These will be used as needed based on the embedding model type
w2v_params = {
    "high_sd_cutoff": 3,    # Filter out words with frequency > mean + 3*std
    "low_n_cutoff": 1,      # Filter out words occurring < 1 times
    "save_vocab": True      # Save vocabulary lists to output directory
}

lexsyn_params = {
    "max_ngram": 3,
    "ignore_duplicates": True,
    "add_stanford_tags": True
}

# Parameters for any embedding model
common_params = {
    "lag": 1
}

# # 1. First, analyze the real conversations
# print(f"Analyzing real conversations with {embedding_model} model...")
# real_results = analyzer.analyze_folder(
#     folder_path=data_path,
#     output_directory=os.path.join(output_folder, "real"),
#     **common_params,
#     **w2v_params,
#     **lexsyn_params
# )

# 2. Next, generate surrogate pairs and analyze them
print(f"Generating and analyzing surrogate pairs with {embedding_model} model...")
baseline_results = analyzer.analyze_baseline(
    input_files=data_path,
    output_directory=os.path.join(output_folder, "baseline"),
    surrogate_directory=os.path.join(output_folder, "surrogates"),
    all_surrogates=False,
    keep_original_turn_order=True,
    # Adjust these to match your actual filename format
    id_separator="_",              # The separator between components
    condition_label="ExpBlock",    # The string that identifies conditions
    dyad_label="ASU-",             # The string that identifies dyads
    # Include all other parameters
    **common_params,
    **w2v_params,
    **lexsyn_params
)


