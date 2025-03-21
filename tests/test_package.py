import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment_cache import SemanticAlignmentAnalyzer

# Initialize analyzer with simplified parameters
analyzer = SemanticAlignmentAnalyzer()

# Get absolute path to the data folder
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "my_package", "data", "prepped_stan_small")

print(f"Looking for data in: {data_path}")

# Process files in the folder
results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory="tests/results"
)

# Inspect results
print(f"\nProcessed {len(results)} rows")
if not results.empty:
    print(f"Columns: {results.columns.tolist()}")
else:
    print("No results were generated. Check the error messages above.")