"""
Basic Usage Example for ALIGN Package

This script demonstrates the basic usage of the ALIGN package
to analyze semantic alignment using BERT embeddings.
"""

import os
import sys

# Add parent directory to path to allow importing from my_package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import LinguisticAlignment

# Define paths for input data and output
# Using the sample data provided with the package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "my_package", "data", "prepped_stan_small")
output_folder = os.path.join(project_root, "examples", "output")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the analyzer with BERT
analyzer = LinguisticAlignment(
    alignment_type="bert",  # Use BERT for semantic alignment
    cache_dir=os.path.join(output_folder, "cache")  # Cache BERT model
)

# Run the analysis
print(f"Analyzing conversations in: {data_path}")
print(f"Results will be saved to: {output_folder}")

results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,
    lag=1  # Analyze alignment with lag of 1 turn
)

print("\nAnalysis complete!")
print(f"Check results in: {output_folder}")
print(f"Number of rows in results: {len(results)}")

# Print a sample of the results
if not results.empty:
    print("\nSample of results (first 5 rows):")
    print(results.head())
    
    # Print available columns
    print("\nAvailable metrics:")
    for col in results.columns:
        if "cosine" in col:
            print(f"- {col}")
else:
    print("No results were generated. Check your data files and parameters.")