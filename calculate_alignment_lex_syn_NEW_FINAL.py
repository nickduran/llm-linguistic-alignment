import os
import ast
import pandas as pd
from math import sqrt
from collections import Counter, OrderedDict
from nltk import ngrams
from tqdm import tqdm

def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing {x}: {e}")
        return []

# Function to calculate cosine similarity
def get_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)

    return float(numerator) / denominator if denominator else 0.0

def compute_ngrams(sequence1, sequence2, ngram_size=2, ignore_duplicates=True, is_tagged=False):
    """
    Computes n-grams for two sequences.
    Handles both lexical (words) and POS (tagged tokens) sequences based on `is_tagged`.
    """
    # Generate n-grams for both sequences
    sequence1_ngrams = list(ngrams(sequence1, ngram_size))
    sequence2_ngrams = list(ngrams(sequence2, ngram_size))

    if is_tagged:
        # POS analysis
        # Remove exact matches if ignore_duplicates is True
        if ignore_duplicates:
            unique_sequence1_ngrams = [ngram for ngram in sequence1_ngrams if ngram not in sequence2_ngrams]
            unique_sequence2_ngrams = [ngram for ngram in sequence2_ngrams if ngram not in sequence1_ngrams]
        else:
            unique_sequence1_ngrams = sequence1_ngrams
            unique_sequence2_ngrams = sequence2_ngrams

        # Extract POS tags only for POS analysis
        pos_sequence1 = [tuple(pos for _, pos in ngram) for ngram in unique_sequence1_ngrams]
        pos_sequence2 = [tuple(pos for _, pos in ngram) for ngram in unique_sequence2_ngrams]

        # Count occurrences of each POS n-gram
        pos_sequence1_count = Counter(pos_sequence1)
        pos_sequence2_count = Counter(pos_sequence2)

        # Return counts for POS analysis
        return None, None, pos_sequence1_count, pos_sequence2_count
    else:
        # Lexical analysis
        # Join words in each n-gram into a string
        lexical_sequence1 = [' '.join(ngram) for ngram in sequence1_ngrams]
        lexical_sequence2 = [' '.join(ngram) for ngram in sequence2_ngrams]

        # Count occurrences of each lexical n-gram
        lexical_sequence1_count = Counter(lexical_sequence1)
        lexical_sequence2_count = Counter(lexical_sequence2)

        # Return counts for lexical analysis
        return lexical_sequence1_count, lexical_sequence2_count, None, None

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

def process_file(file_path, max_ngram=2, delay=1, ignore_duplicates=True, add_stanford_tags=False):
    """Processes a single file to calculate lexical and POS n-grams and alignment metrics."""
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    if df.empty or len(df) <= 1:
        print(f"Skipping invalid file: {file_path}")
        return pd.DataFrame()  # Skip invalid files

    # Convert string representations of lists/tuples to actual lists/tuples using safe_literal_eval
    for col in ['token', 'lemma', 'tagged_token', 'tagged_lemma', 'tagged_stan_token', 'tagged_stan_lemma']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)

    # Add 'content' to the list of columns to lag
    df = pair_and_lag_columns(df, ['token', 'lemma', 'tagged_token', 'tagged_lemma', 'content'] +
                                   (['tagged_stan_token', 'tagged_stan_lemma'] if add_stanford_tags else []))

    # Initialize time counter
    time_counter = 1

    # Calculate n-grams and cosine similarity
    alignment_data = []
    condition_info = os.path.basename(file_path)  # Extract file name once
    for _, row in df.iterrows():
        results = OrderedDict()

        # Add 'time' as the first entry
        results['time'] = time_counter
        time_counter += 1  # Increment time counter

        # Perform n-gram calculations and add results
        for n in range(1, max_ngram + 1):
            # Lexical analysis
            lexical_tok1_count, lexical_tok2_count, _, _ = compute_ngrams(
                row['token1'], row['token2'], ngram_size=n, ignore_duplicates=False, is_tagged=False)
            lexical_lem1_count, lexical_lem2_count, _, _ = compute_ngrams(
                row['lemma1'], row['lemma2'], ngram_size=n, ignore_duplicates=False, is_tagged=False)
            results[f'lexical_tok{n}_cosine'] = get_cosine_similarity(lexical_tok1_count, lexical_tok2_count)
            results[f'lexical_lem{n}_cosine'] = get_cosine_similarity(lexical_lem1_count, lexical_lem2_count)

            # POS analysis
            _, _, pos_tok1_count, pos_tok2_count = compute_ngrams(
                row['tagged_token1'], row['tagged_token2'], ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
            _, _, pos_lem1_count, pos_lem2_count = compute_ngrams(
                row['tagged_lemma1'], row['tagged_lemma2'], ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
            results[f'pos_tok{n}_cosine'] = get_cosine_similarity(pos_tok1_count, pos_tok2_count)
            results[f'pos_lem{n}_cosine'] = get_cosine_similarity(pos_lem1_count, pos_lem2_count)

            # Optional Stanford POS tags (if applicable)
            if add_stanford_tags:
                _, _, stan_tok1_count, stan_tok2_count = compute_ngrams(
                    row['tagged_stan_token1'], row['tagged_stan_token2'], ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
                _, _, stan_lem1_count, stan_lem2_count = compute_ngrams(
                    row['tagged_stan_lemma1'], row['tagged_stan_lemma2'], ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
                results[f'stan_pos_tok{n}_cosine'] = get_cosine_similarity(stan_tok1_count, stan_tok2_count)
                results[f'stan_pos_lem{n}_cosine'] = get_cosine_similarity(stan_lem1_count, stan_lem2_count)

        # Add 'utter_order'
        results['utter_order'] = row['utter_order']

        # Add columns you want at the end
        # Calculate utterance lengths
        results['utterance_length1'] = len(row['token1']) if row['token1'] else 0
        results['utterance_length2'] = len(row['token2']) if row['token2'] else 0

        # Add condition_info
        results['condition_info'] = condition_info

        # Optionally, include the actual utterance content
        results['content1'] = row['content1']
        results['content2'] = row['content2']

        alignment_data.append(results)

    # Convert results to DataFrame
    df_results = pd.DataFrame(alignment_data)

    # Reorder columns to move specified columns to the end
    cols_to_move_to_end = ['utterance_length1', 'utterance_length2', 'condition_info', 'content1', 'content2']
    cols = ['time'] + [col for col in df_results.columns if col not in cols_to_move_to_end + ['time']] + cols_to_move_to_end
    df_results = df_results[cols]

    # Return the DataFrame
    return df_results

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




# Example call to process_file
file_path = 'data/prepped_stan_small/ASU-T10_ExpBlock1-Oneatatime.txt'
df_alignment = process_file(file_path, max_ngram=2, delay=1, ignore_duplicates=True, add_stanford_tags=False)

# Display the output DataFrame
print(df_alignment.head())



# Define the folder path
folder_path = "data/prepped_stan_small"

# Run the processing function on all files in the specified folder
final_df = process_all_files(folder_path, max_ngram=2, delay=1, ignore_duplicates=False, add_stanford_tags=True)

# Preview the resulting DataFrame
print(final_df.head())

# Save the output to a CSV file
final_df.to_csv("output_alignment_results_ignoreFALSE.csv", index=False)




