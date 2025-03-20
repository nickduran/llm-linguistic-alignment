# test_package.py
from my_package.alignment import SemanticAlignmentAnalyzer

# Initialize the analyzer
# The cache file will be created if it doesn't exist
analyzer = SemanticAlignmentAnalyzer(
    model_name="bert-base-uncased",
    cache_path="./bert_embedding_cache.pkl"  # This will be created if it doesn't exist
)

# Path to your conversation files
folder_path = "./my_package/data/prepped_stan_small"  # Replace with your actual folder path
output_directory = "./my_package/data"      # Replace with where you want results saved

# Run the analysis - this will:
# 1. Create the cache file if it doesn't exist
# 2. Calculate new embeddings and update the cache
# 3. Save the results to the output directory
result_df = analyzer.analyze_folder(
    folder_path=folder_path,
    output_directory=output_directory
)

# # Show a sample of the results
# print(result_df.head())



