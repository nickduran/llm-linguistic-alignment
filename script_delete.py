# ===============================
# IMPORTS
# ===============================
import os
from os import listdir, path
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import re
import requests
import ast
from tqdm import tqdm  # for progress bars
from sklearn.metrics.pairwise import cosine_similarity
import gensim 
import gensim.downloader as api 
from gensim.models import KeyedVectors 

# ===============================
# SETUP
# ===============================
# Determine the current working directory
script_dir = os.getcwd()

# Define the local cache directory relative to the current working directory
local_cache_dir = os.path.join(script_dir, "gensim-data")
os.makedirs(local_cache_dir, exist_ok=True)
print(f"Local cache directory: {local_cache_dir}")

# Set the BASE_DIR for gensim data
api.BASE_DIR = local_cache_dir
print(f"Gensim BASE_DIR set to: {api.BASE_DIR}")

# Function to download and cache models
def download_and_cache_models(models, cache_dir):
    api.BASE_DIR = cache_dir
    for model_name in models:
        model_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(model_path):
            try:
                print(f"Downloading model: {model_name}")
                model = api.load(model_name)
                print(f"Downloaded and cached model: {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
        else:
            print(f"Model {model_name} already exists at: {model_path}")

# List of models to download and cache
models_to_cache = ['word2vec-google-news-300', 'glove-twitter-200']
download_and_cache_models(models_to_cache, local_cache_dir)

# Function to load models if they are not already loaded
def load_model_if_not_exists(model_path, binary=True):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# Load the Google News model if it is not already loaded
if 'w2v_google_model' not in globals():
    w2v_google_model_path = os.path.join(local_cache_dir, 'word2vec-google-news-300', 'word2vec-google-news-300.gz')
    w2v_google_model = load_model_if_not_exists(w2v_google_model_path, binary=True)
    if w2v_google_model is not None:
        print("Word2Vec Google News model loaded from local cache successfully.")
    else:
        print("Failed to load Word2Vec Google News model.")

# ===============================
# FUNCTION DEFINITIONS
# ===============================
# Function to aggregate conversations
def aggregate_conversations(folder_path: str) -> pd.DataFrame:
    text_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
    concatenated_df = pd.DataFrame()

    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
    
    return concatenated_df

# Function to build filtered vocabulary from aggegrated conversations
def build_filtered_vocab(data: pd.DataFrame, output_file_directory: str, high_sd_cutoff: float = 3, low_n_cutoff: int = 1):
    # Tokenize the lemmas
    all_sentences = [re.sub(r'[^\w\s]+', '', str(row)).split() for row in data['lemma']]
    all_words = [word for sentence in all_sentences for word in sentence]
    
    # Frequency count using Counter
    frequency = Counter(all_words)
    
    # Filter out one-letter words and those below low_n_cutoff
    frequency_filt = {word: freq for word, freq in frequency.items() if len(word) > 1 and freq > low_n_cutoff}
    
    # Remove high-frequency words if high_sd_cutoff is specified
    if high_sd_cutoff is not None:
        mean_freq = np.mean(list(frequency_filt.values()))
        std_freq = np.std(list(frequency_filt.values()))
        cutoff_freq = mean_freq + (std_freq * high_sd_cutoff)
        filteredWords = {word: freq for word, freq in frequency_filt.items() if freq < cutoff_freq}
    else:
        filteredWords = frequency_filt
    
    # Convert to DataFrames for exporting
    vocabfreq_all = pd.DataFrame(frequency.items(), columns=["word", "count"]).sort_values(by='count', ascending=False)
    vocabfreq_filt = pd.DataFrame(filteredWords.items(), columns=["word", "count"]).sort_values(by='count', ascending=False)
    
    # Save to files
    vocabfreq_all.to_csv(os.path.join(output_file_directory, 'vocab_unfilt_freqs.txt'), encoding='utf-8', index=False, sep='\t')
    vocabfreq_filt.to_csv(os.path.join(output_file_directory, 'vocab_filt_freqs.txt'), encoding='utf-8', index=False, sep='\t')
    
    return list(frequency.keys()), list(filteredWords.keys())

# Function to check if a column contains list-like strings
def is_list_like_column(series):
    try:
        return series.apply(lambda x: x.strip().startswith("[")).all()
    except AttributeError:
        return False

# Function to convert columns with list-like strings to actual lists
def convert_columns_to_lists(df: pd.DataFrame) -> pd.DataFrame:
    columns_converted = []
    for col in df.columns:
        if is_list_like_column(df[col]):
            df[col] = df[col].apply(ast.literal_eval)
            columns_converted.append(col)
    return df, columns_converted

# Function to get lagged conversational turns and restructure dataframe
def process_input_data(df: pd.DataFrame, include_stan: bool = True) -> pd.DataFrame:
    # Base columns to lag
    columns_to_lag = ['content', 'token', 'lemma', 'tagged_token', 'tagged_lemma']
    
    # Optionally include "stan" columns if they exist
    if include_stan:
        stan_columns = [col for col in df.columns if 'stan' in col]
        columns_to_lag.extend(stan_columns)
    
    for col in columns_to_lag:
        if col in df.columns:  # Ensure the column exists in the DataFrame
            df[f'{col}1'] = df[col]
            df[f'{col}2'] = df[col].shift(-1)
    
    df['utter_order'] = df['participant'] + ' ' + df['participant'].shift(-1)
    
    return df

# Main function to process the file
def process_file(file_path, large_list: list):       
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    
    # Convert columns with list-like strings to actual lists
    df, columns_converted = convert_columns_to_lists(df)

    # Filtering based on user-specified requests
    columns_to_filter = ['lemma','token']
    for col in columns_to_filter:
        # First filter: Keep only words in filter_model
        df[col] = df[col].apply(lambda token_list: [word for word in token_list if word in large_list])
    
    # Do the lagging
    df = process_input_data(df)

    return df

# Function to compute cosine similarities between embeddings
def compute_cosine_similarities(df: pd.DataFrame, columns: list):
    for col1, col2 in columns:
        similarities = []
        for i in range(len(df)):
            vec1 = df.iloc[i][col1]
            vec2 = df.iloc[i][col2]
            if vec1 is not None and vec2 is not None:
                similarity = cosine_similarity([vec1], [vec2])[0][0]
            else:
                similarity = None
            similarities.append(similarity)

        # Determine whether this is for "token" or "lemma" based on the column name
        if 'token' in col1:
            similarity_column_name = 'token_cosine_similarity'
        elif 'lemma' in col1:
            similarity_column_name = 'lemma_cosine_similarity'

        df[similarity_column_name] = similarities
    return df

# Function to get the sum embeddings for each list of tokens
def get_sum_embeddings(token_list, model):
    if token_list is None:
        return None    
    embeddings = []
    for word in token_list:
        if word in model.key_to_index:  # Check if word is in the model vocabulary
            embeddings.append(model[word])    
    if embeddings:
        sum_embedding = np.sum(embeddings, axis=0)
        return sum_embedding
    else:
        return None  # Or handle empty embeddings as you see fit

# ===============================
# MAIN SCRIPT
# ===============================
if __name__ == "__main__":
    # Path to the folder containing the text files
    folder_path = "./data/prepped_stan_small"
    output_file_directory = "output"
    text_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]

    # Aggregate individual conversation files
    concatenated_text_files = aggregate_conversations(folder_path)
        
    # Build filtered vocabulary from aggregated data
    vocab_all, vocab_filtered = build_filtered_vocab(concatenated_text_files, output_file_directory)

    # Process each file and update the cache
    concatenated_df = pd.DataFrame()
    for file_name in tqdm(text_files, desc="Processing files"):
        file_path = os.path.join(folder_path, file_name)
        df = process_file(file_path, vocab_filtered)

        # Create columns of embeddings
        for column in ["lemma", "token"]:
            df[f"{column}1_sum_embedding"] = df[f"{column}1"].apply(lambda tokens: get_sum_embeddings(tokens, w2v_google_model))
            df[f"{column}2_sum_embedding"] = df[f"{column}2"].apply(lambda tokens: get_sum_embeddings(tokens, w2v_google_model))

        # Columns to compute similarities
        embedding_columns = [
            ("lemma1_sum_embedding", "lemma2_sum_embedding"),
            ("token1_sum_embedding", "token2_sum_embedding")
        ]

        # Compute cosine similarities
        df = compute_cosine_similarities(df, embedding_columns)
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)


### Explanation

# 1. **IMPORTS**:
#     - All the import statements are grouped together at the top.

# 2. **SETUP**:
#     - The section sets up the local cache directory and downloads necessary models.

# 3. **FUNCTION DEFINITIONS**:
#     - All the functions are defined in this section. The order is adjusted for better readability:
#         - `aggregate_conversations`
#         - `build_filtered_vocab`
#         - `is_list_like_column`
#         - `convert_columns_to_lists`
#         - `process_input_data`
#         - `process_file`
#         - `compute_cosine_similarities`
#         - `get_sum_embeddings`

# 4. **MAIN SCRIPT**:
#     - The main script logic is contained within the `if __name__ == "__main__":` block to ensure it runs only when the script is executed directly. This includes the data processing steps and function calls.

# This structure makes the code more organized and easier to understand.