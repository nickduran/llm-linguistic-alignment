import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import either specific analyzers or the unified interface
from my_package.alignment import SemanticAlignment
# OR
# from my_package.alignment_bert import SemanticAlignmentAnalyzer
# from my_package.alignment_w2v import SemanticAlignmentW2V

# Option 1: Using the unified interface
# analyzer = SemanticAlignment(embedding_model="bert")
analyzer = SemanticAlignment(embedding_model="word2vec")

# Option 2: Using the specific Word2Vec analyzer
# analyzer = SemanticAlignmentW2V(model_name="word2vec-google-news-300")

# # Get path to data folder
# data_path = "path/to/conversation/files"

# Get absolute path to the data folder
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_small")

print(f"Looking for data in: {data_path}")

# Set the lag parameter
lag = 1  # Use lag of 2 for testing

# Process files in the folder with lag parameter
results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory="tests/results",
    lag=lag
)

# Inspect results
print(f"\nProcessed {len(results)} rows with lag {lag}")
if not results.empty:
    print(f"Columns: {results.columns.tolist()}")
else:
    print("No results were generated. Check the error messages above.")
