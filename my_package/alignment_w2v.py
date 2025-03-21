# my_package/alignment_w2v.py
import os
import numpy as np
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .word2vec_model import Word2VecWrapper

class SemanticAlignmentW2V:
    def __init__(self, model_name="word2vec-google-news-300", cache_dir=None, token=None):
        """
        Initialize the semantic alignment analyzer with Word2Vec
        
        Args:
            model_name: Name of the Word2Vec model to use
            cache_dir: Directory to cache models (optional)
            token: Hugging Face token (optional)
        """
        self.w2v_wrapper = Word2VecWrapper(model_name, cache_dir, token)
        self.model_name = model_name.split('/')[-1]  # Extract model name for column naming
        self.vocabulary = None
        
    def build_vocabulary(self, data, high_sd_cutoff=3, low_n_cutoff=1):
        """
        Build a filtered vocabulary from the data
        
        Args:
            data: DataFrame containing text data
            high_sd_cutoff: Standard deviation cutoff for high-frequency words
            low_n_cutoff: Minimum frequency cutoff
            
        Returns:
            list: Filtered vocabulary
        """
        # Extract words from the 'lemma' column if it exists, otherwise use 'content'
        column = 'lemma' if 'lemma' in data.columns else 'content'
        
        # Convert string representations of lists to actual lists if needed
        if data[column].dtype == 'object' and data[column].str.startswith('[').any():
            import ast
            all_words = [word for row in data[column] for word in ast.literal_eval(row) if isinstance(row, str)]
        else:
            all_sentences = [re.sub(r'[^\w\s]+', '', str(row)).split() for row in data[column]]
            all_words = [word for sentence in all_sentences for word in sentence]
        
        # Count word frequencies
        frequency = Counter(all_words)
        
        # Apply low frequency filter
        frequency_filt = {word: freq for word, freq in frequency.items() if len(word) > 1 and freq > low_n_cutoff}
        
        # Apply high frequency filter if specified
        if high_sd_cutoff is not None:
            mean_freq = np.mean(list(frequency_filt.values()))
            std_freq = np.std(list(frequency_filt.values()))
            cutoff_freq = mean_freq + (std_freq * high_sd_cutoff)
            filtered_words = {word: freq for word, freq in frequency_filt.items() if freq < cutoff_freq}
        else:
            filtered_words = frequency_filt
            
        self.vocabulary = list(filtered_words.keys())
        return self.vocabulary
        
    def get_embedding(self, tokens):
        """
        Get Word2Vec embedding for tokens
        
        Args:
            tokens: List of tokens to encode
            
        Returns:
            numpy.ndarray: Embedding vector or None if no tokens can be encoded
        """
        if tokens is None or not tokens:
            return None
            
        # Filter tokens by vocabulary if available
        if self.vocabulary:
            tokens = [token for token in tokens if token in self.vocabulary]
            
        return self.w2v_wrapper.get_text_embedding(tokens)
    
    def pair_and_lag_columns(self, df, columns_to_lag, suffix1='1', suffix2='2', lag=1):
        """
        Creates lagged pairs of specified columns
        
        Args:
            df: DataFrame to process
            columns_to_lag: List of column names to lag
            suffix1: Suffix for original columns
            suffix2: Suffix for lagged columns
            lag: Number of rows to lag (default: 1)
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        for col in columns_to_lag:
            if col in df.columns:
                df[f'{col}{suffix1}'] = df[col]
                df[f'{col}{suffix2}'] = df[col].shift(-lag)
        
        # Add participant order information
        if 'participant' in df.columns:
            df['utter_order'] = df['participant'] + ' ' + df['participant'].shift(-lag)
        
        return df
        
    def calculate_cosine_similarity(self, df, embedding_pairs):
        """
        Computes cosine similarities between pairs of vectors
        
        Args:
            df: DataFrame containing embeddings
            embedding_pairs: List of (col1, col2) tuples to compare
            
        Returns:
            pd.DataFrame: DataFrame with similarity columns added
        """
        for col1, col2 in embedding_pairs:
            similarities = []
            
            for _, row in df.iterrows():
                # Check if both embeddings exist and are not None
                if isinstance(row.get(col1), np.ndarray) and isinstance(row.get(col2), np.ndarray):
                    embed1 = row[col1]
                    embed2 = row[col2]
                    
                    # Check if embeddings are not empty
                    if embed1.size > 0 and embed2.size > 0:
                        try:
                            sim = cosine_similarity(
                                embed1.reshape(1, -1), 
                                embed2.reshape(1, -1)
                            )[0][0]
                            similarities.append(sim)
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                            similarities.append(None)
                    else:
                        similarities.append(None)
                else:
                    similarities.append(None)
            
            similarity_column_name = f"{col1}_{col2}_cosine_similarity"
            df[similarity_column_name] = similarities
        
        return df
    
    def convert_list_columns(self, df):
        """
        Convert string representation of lists to actual lists
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        import ast
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check if column contains string representations of lists
                    if df[col].str.startswith('[').any() and df[col].str.endswith(']').any():
                        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
                except:
                    pass
                    
        return df
    
    def process_file(self, file_path, lag=1):
        """
        Process a single file to compute embeddings and similarities
        
        Args:
            file_path: Path to the file to process
            lag: Number of turns to lag when pairing utterances (default: 1)
            
        Returns:
            pd.DataFrame: Processed DataFrame with embeddings and similarities
        """
        try:
            # Read the file
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            
            # Check required columns
            if 'content' not in df.columns:
                print(f"Warning: 'content' column not found in {file_path}")
                print(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Convert list columns if needed
            df = self.convert_list_columns(df)
            
            # Pair and lag columns - ensure participant exists
            if 'participant' in df.columns:
                df = self.pair_and_lag_columns(df, columns_to_lag=['content', 'lemma', 'token'], lag=lag)
            else:
                print(f"Warning: 'participant' column not found in {file_path}, skipping participant tracking")
                # Create lagged columns without participant tracking
                for col in ['content', 'lemma', 'token']:
                    if col in df.columns:
                        df[f'{col}1'] = df[col]
                        df[f'{col}2'] = df[col].shift(-lag)
            
            # Save original filename for reference
            df['source_file'] = os.path.basename(file_path)
            
            # Add lag information to the dataframe
            df['lag'] = lag
            
            # Compute embeddings
            print(f"Computing embeddings for {file_path}...")
            embedding_columns = []
            
            # Process content embeddings
            for column in ["content1", "content2"]:
                if column in df.columns:
                    col_name = f"{column}_embedding_{self.model_name}"
                    df[col_name] = None  # Initialize with None
                    
                    # Process row by row
                    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {column}"):
                        if pd.notna(row[column]) and isinstance(row[column], str):
                            # Tokenize the content
                            tokens = row[column].lower().split()
                            
                            # Get the embedding
                            embedding = self.get_embedding(tokens)
                            
                            # Store the embedding
                            if embedding is not None:
                                df.at[idx, col_name] = embedding
                                df.at[idx, f"{col_name}_dims"] = embedding.shape[0]
                    
                    embedding_columns.append(col_name)
            
            # Process token or lemma embeddings if available
            for base_col in ["token", "lemma"]:
                for suffix in ["1", "2"]:
                    column = f"{base_col}{suffix}"
                    if column in df.columns:
                        col_name = f"{column}_embedding_{self.model_name}"
                        df[col_name] = None  # Initialize with None
                        
                        # Process row by row
                        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {column}"):
                            tokens = row[column]
                            if tokens is not None and (isinstance(tokens, list) or (isinstance(tokens, str) and tokens.startswith('['))):
                                # Convert string representation to list if needed
                                if isinstance(tokens, str) and tokens.startswith('['):
                                    import ast
                                    tokens = ast.literal_eval(tokens)
                                
                                # Get the embedding
                                embedding = self.get_embedding(tokens)
                                
                                # Store the embedding
                                if embedding is not None:
                                    df.at[idx, col_name] = embedding
                                    df.at[idx, f"{col_name}_dims"] = embedding.shape[0]
                        
                        embedding_columns.append(col_name)
            
            # Calculate cosine similarities if we have embeddings
            if len(embedding_columns) >= 2:
                embedding_pairs = [(embedding_columns[0], embedding_columns[1])]
                df = self.calculate_cosine_similarity(df, embedding_pairs)
            
            return df
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for debugging
            return pd.DataFrame()  # Return empty dataframe on error
    
    def analyze_folder(self, folder_path, output_directory=None, file_pattern="*.txt", lag=1, build_vocab=True):
        """
        Analyze semantic alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Directory to save results (optional)
            file_pattern: Pattern to match text files (default: "*.txt")
            lag: Number of turns to lag when pairing utterances (default: 1)
            build_vocab: Whether to build a vocabulary from all files (default: True)
            
        Returns:
            pd.DataFrame: Concatenated results for all files
        """
        import glob
        result_df = pd.DataFrame()
        
        # Get list of files matching pattern
        file_paths = glob.glob(os.path.join(folder_path, file_pattern))
        
        if not file_paths:
            print(f"No files matching pattern '{file_pattern}' found in {folder_path}")
            return result_df
        
        print(f"Found {len(file_paths)} files to process with lag {lag}")
        
        # Build vocabulary from all files if requested
        if build_vocab:
            print("Building vocabulary from all files...")
            all_data = pd.DataFrame()
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                    all_data = pd.concat([all_data, df], ignore_index=True)
                except Exception as e:
                    print(f"Error reading {file_path} for vocabulary building: {e}")
            
            if not all_data.empty:
                self.build_vocabulary(all_data)
                print(f"Built vocabulary with {len(self.vocabulary)} words")
            else:
                print("Warning: Could not build vocabulary from files")
        
        # Process each file
        successful_files = 0
        for file_path in tqdm(file_paths, desc=f"Processing files with {self.model_name}"):
            try:
                # Pass the lag parameter to process_file
                file_df = self.process_file(file_path, lag=lag)
                if not file_df.empty:
                    result_df = pd.concat([result_df, file_df], ignore_index=True)
                    successful_files += 1
            except Exception as e:
                print(f"Fatal error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Successfully processed {successful_files} out of {len(file_paths)} files")
        
        # Rename the similarity columns for consistency
        for col in result_df.columns:
            if '_cosine_similarity' in col:
                new_col_name = f"{self.model_name}_cosine_similarity"
                result_df = result_df.rename(columns={col: new_col_name})
                break
        
        # Save results if output directory is provided
        if output_directory and not result_df.empty:
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, f"semantic_alignment_{self.model_name}_lag{lag}.csv")
            result_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return result_df