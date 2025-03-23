import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import either specific analyzers or the unified interface
from my_package.alignment import SemanticAlignment
# OR
# from my_package.alignment_bert import SemanticAlignmentAnalyzer
# from my_package.alignment_w2v import SemanticAlignmentW2V

# ORIGINAL Option 1: Using the unified interface
# analyzer = SemanticAlignment(embedding_model="bert")
# analyzer = SemanticAlignment(embedding_model="word2vec")

# ORIGINAL Option 2: Using the specific Word2Vec analyzer
# analyzer = SemanticAlignmentW2V(model_name="word2vec-google-news-300")

# # Get path to data folder
# data_path = "path/to/conversation/files"

# Get absolute path to the data folder
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_small")

print(f"Looking for data in: {data_path}")

# Option 1: BERT with default settings (Word2Vec-specific parameters are ignored)
# bert_analyzer = SemanticAlignment(embedding_model="bert")
# bert_results = bert_analyzer.analyze_folder(
#     folder_path=data_path,
#     output_directory="tests/results/bert",
#     lag=1
# )

# Option 2: Word2Vec with customized vocabulary filtering
w2v_analyzer = SemanticAlignment(embedding_model="word2vec")
w2v_results = w2v_analyzer.analyze_folder(
    folder_path=data_path,
    output_directory="tests/results/word2vec",
    lag=1,
    high_sd_cutoff=2.5,  # More aggressive filtering of high-frequency words
    low_n_cutoff=1,      # Filter out words occurring less than twice
    save_vocab=True      # Save vocabulary lists to output directory
)

# Compare results
# print(f"BERT results: {len(bert_results)} rows")
print(f"Word2Vec results: {len(w2v_results)} rows")




# # Set the lag parameter
# lag = 1  # Use lag of 2 for testing

# # Process files in the folder with lag parameter
# results = analyzer.analyze_folder(
#     folder_path=data_path,
#     output_directory="tests/results",
#     lag=lag
# )

# # Inspect results
# print(f"\nProcessed {len(results)} rows with lag {lag}")
# if not results.empty:
#     print(f"Columns: {results.columns.tolist()}")
# else:
#     print("No results were generated. Check the error messages above.")
