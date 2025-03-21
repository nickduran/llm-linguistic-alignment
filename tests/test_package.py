# tests/test_package_with_lag.py
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new module
from my_package.alignment import SemanticAlignmentAnalyzer

# Initialize analyzer
analyzer = SemanticAlignmentAnalyzer()

# Get absolute path to the data folder
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_small")

print(f"Looking for data in: {data_path}")

# Set the lag parameter
lag = 2  # Use lag of 2 for testing

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