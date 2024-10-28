import os
import re
import pandas as pd
import ast  # Use ast.literal_eval for safe conversion of string representations to lists and tuples
from collections import Counter, OrderedDict
from nltk.util import ngrams
from math import sqrt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate cosine similarity
def get_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)
    return float(numerator) / denominator if denominator else 0.0

# Function to process n-grams for lexical and POS
def compute_ngrams(sequence1, sequence2, ngram_size=2, ignore_duplicates=True):
    """Computes n-grams and optionally removes duplicates between two sequences."""
    sequence1_ngrams = set(ngrams(sequence1, ngram_size))
    sequence2_ngrams = set(ngrams(sequence2, ngram_size))
    
    if ignore_duplicates:
        sequence1_ngrams -= sequence2_ngrams
        sequence2_ngrams -= sequence1_ngrams

    sequence1_count = Counter(sequence1_ngrams)
    sequence2_count = Counter(sequence2_ngrams)
    
    # Debug prints
    print("Sequence 1 n-grams:", list(sequence1_ngrams))
    print("Sequence 2 n-grams:", list(sequence2_ngrams))

    return sequence1_count, sequence2_count

# Function to pair and lag specified columns
def pair_and_lag_columns(df, columns_to_lag, suffix1='1', suffix2='2'):
    """Pairs and lags specified columns, creating new columns with designated suffixes."""
    for col in columns_to_lag:
        if col in df.columns:
            df[f'{col}{suffix1}'] = df[col]
            df[f'{col}{suffix2}'] = df[col].shift(-1)
    df['utter_order'] = df['participant'] + ' ' + df['participant'].shift(-1)
    return df.dropna(subset=[f"{col}{suffix2}" for col in columns_to_lag])

# Function to calculate cosine similarities for specific column pairs
def calculate_cosine_similarity(df, embedding_pairs):
    """Computes cosine similarity for specified pairs of columns and adds results as new columns."""
    for col1, col2 in embedding_pairs:
        df[f"{col1}_{col2}_cosine_similarity"] = df.apply(
            lambda row: get_cosine_similarity(row[col1], row[col2])
            if row[col1] is not None and row[col2] is not None else None,
            axis=1
        )
    return df

# Core function to process a single file
def process_file(file_path, max_ngram=2, delay=1, ignore_duplicates=True, add_stanford_tags=False):
    """Processes a single file to calculate lexical and POS n-grams and alignment metrics."""
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    if df.empty or len(df) <= 1:
        print(f"Skipping invalid file: {file_path}")
        return pd.DataFrame()  # Skip invalid files

    # Convert string representations of lists/tuples to actual lists/tuples
    for col in ['token', 'lemma', 'tagged_token', 'tagged_lemma', 'tagged_stan_token', 'tagged_stan_lemma']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Lag specified columns for comparison
    df = pair_and_lag_columns(df, ['token', 'lemma', 'tagged_token', 'tagged_lemma'] +
                                   (['tagged_stan_token', 'tagged_stan_lemma'] if add_stanford_tags else []))

    # Calculate n-grams and cosine similarity
    alignment_data = []
    for _, row in df.iterrows():
        results = OrderedDict()
        for n in range(1, max_ngram + 1):
            # Lexical alignment
            tok1_count, tok2_count = compute_ngrams(row['token1'], row['token2'], n, ignore_duplicates)
            lem1_count, lem2_count = compute_ngrams(row['lemma1'], row['lemma2'], n, ignore_duplicates)
            results[f'lexical_tok{n}_cosine'] = get_cosine_similarity(tok1_count, tok2_count)
            results[f'lexical_lem{n}_cosine'] = get_cosine_similarity(lem1_count, lem2_count)

            # POS alignment
            pos_tok1_count, pos_tok2_count = compute_ngrams(row['tagged_token1'], row['tagged_token2'], n, ignore_duplicates)
            pos_lem1_count, pos_lem2_count = compute_ngrams(row['tagged_lemma1'], row['tagged_lemma2'], n, ignore_duplicates)
            results[f'pos_tok{n}_cosine'] = get_cosine_similarity(pos_tok1_count, pos_tok2_count)
            results[f'pos_lem{n}_cosine'] = get_cosine_similarity(pos_lem1_count, pos_lem2_count)

            # Optional Stanford POS tags
            if add_stanford_tags:
                stan_tok1_count, stan_tok2_count = compute_ngrams(row['tagged_stan_token1'], row['tagged_stan_token2'], n, ignore_duplicates)
                stan_lem1_count, stan_lem2_count = compute_ngrams(row['tagged_stan_lemma1'], row['tagged_stan_lemma2'], n, ignore_duplicates)
                results[f'stan_pos_tok{n}_cosine'] = get_cosine_similarity(stan_tok1_count, stan_tok2_count)
                results[f'stan_pos_lem{n}_cosine'] = get_cosine_similarity(stan_lem1_count, stan_lem2_count)

        results['utter_order'] = row['utter_order']
        alignment_data.append(results)

    # Convert results to DataFrame and return
    return pd.DataFrame(alignment_data)

# Main processing loop for all files in the specified folder
def process_all_files(folder_path, max_ngram=2, delay=1, ignore_duplicates=True, add_stanford_tags=False):
    """Processes all text files in a folder, concatenating alignment results."""
    text_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
    concatenated_df = pd.DataFrame()
    
    for file_name in tqdm(text_files, desc="Processing files"):
        file_path = os.path.join(folder_path, file_name)
        df = process_file(file_path, max_ngram, delay, ignore_duplicates, add_stanford_tags)
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

    return concatenated_df

# Example usage
# folder_path = "data/prepped_stan_small"
# final_df = process_all_files(folder_path, max_ngram=2, delay=1, ignore_duplicates=True, add_stanford_tags=False)
# print(final_df.head())
# final_df.to_csv("output_filename.csv", index=False)


# Define the folder path and get all text files
folder_path = "data/prepped_stan_small"
text_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]

# Check if there are any files in the directory
if text_files:
    # Select the first file only
    file_to_process = os.path.join(folder_path, text_files[0])

    # Process just this single file
    final_df = process_file(file_to_process, max_ngram=2, delay=1, ignore_duplicates=True, add_stanford_tags=False)

    # Display the result or save to CSV if desired
    print(final_df.head())
    final_df.to_csv("output_single_file.csv", index=False)
    print("Processed single file saved to output_single_file.csv")
else:
    print("No files found in the specified folder.")