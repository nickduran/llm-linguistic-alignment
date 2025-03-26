import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import SemanticAlignment
from my_package.surrogates import SurrogateGenerator, SurrogateAlignment

# Define path to data folder - adjust paths as needed for your environment
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan")
output_folder = "tests/results"

# Make sure root output directory exists
os.makedirs(output_folder, exist_ok=True)
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

# 1. First, analyze the real conversations
print(f"Analyzing real conversations with {embedding_model} model...")
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,  # Now uses root directory, analyzer will create model-specific subdir
    **common_params,
    **w2v_params,
    **lexsyn_params
)

# 2. Next, generate surrogate pairs and analyze them
print(f"Generating and analyzing surrogate pairs with {embedding_model} model...")
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
    **w2v_params,
    **lexsyn_params
)

# 3. If you want to compare results, you now need to find them in the model-specific directory
if not real_results.empty and not baseline_results.empty:
    model_dir = os.path.join(output_folder, embedding_model)
    
    # Define comparison function
    def compare_alignment(real_df, baseline_df, metric_columns):
        """Compare real and baseline alignment for specified metrics"""
        comparison_data = []
        
        # Get average values for each metric
        for col in metric_columns:
            if col in real_df.columns and col in baseline_df.columns:
                real_mean = real_df[col].mean()
                baseline_mean = baseline_df[col].mean()
                difference = real_mean - baseline_mean
                percent_increase = (difference / baseline_mean) * 100 if baseline_mean != 0 else float('inf')
                
                comparison_data.append({
                    'Metric': col,
                    'Real_Mean': real_mean,
                    'Baseline_Mean': baseline_mean,
                    'Difference': difference,
                    'Percent_Increase': percent_increase
                })
        
        return pd.DataFrame(comparison_data)

    # Select metrics based on the embedding model
    if embedding_model == "lexsyn":
        metrics = ['lexical_master_cosine', 'syntactic_master_cosine']
        for n in range(1, lexsyn_params["max_ngram"] + 1):  # For each n-gram size
            metrics.extend([
                f'lexical_tok{n}_cosine', 
                f'lexical_lem{n}_cosine',
                f'pos_tok{n}_cosine', 
                f'pos_lem{n}_cosine'
            ])
            if lexsyn_params["add_stanford_tags"]:
                metrics.extend([
                    f'stan_pos_tok{n}_cosine', 
                    f'stan_pos_lem{n}_cosine'
                ])
    elif embedding_model == "bert":
        metrics = ['bert-base-uncased_cosine_similarity']
    elif embedding_model == "word2vec":
        metrics = ['master_word2vec-google-news-300_cosine_similarity']

    # Generate comparison
    comparison_results = compare_alignment(real_results, baseline_results, metrics)
    comparison_path = os.path.join(model_dir, f"alignment_comparison.csv")
    comparison_results.to_csv(comparison_path, index=False)
    print(f"Comparison saved to {comparison_path}")
    
    # Print summary
    print("\nAlignment Comparison Summary:")
    for _, row in comparison_results.iterrows():
        print(f"{row['Metric']}: Real={row['Real_Mean']:.4f}, Baseline={row['Baseline_Mean']:.4f}, " 
              f"Difference={row['Difference']:.4f} ({row['Percent_Increase']:.2f}%)")
else:
    print("Warning: Unable to compare results - one or both result sets are empty")