# my_package/alignment.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from .model import BertWrapper

class SemanticAlignmentAnalyzer:
    def __init__(self, model_name="bert-base-uncased", token=None, cache_path=None):
        """
        Initialize the semantic alignment analyzer
        
        Args:
            model_name: Name of the BERT model to use
            token: Hugging Face token (optional)
            cache_path: Path to embedding cache file (optional)
        """
        self.bert_wrapper = BertWrapper(model_name, token)
        self.tokenizer = self.bert_wrapper.tokenizer
        self.model = self.bert_wrapper.model
        self.model_name = model_name.split('/')[-1]  # Extract model name for column naming
        self.cache_path = cache_path or "bert_embedding_cache.pkl"
        self.embedding_cache = self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from file or initialize empty cache"""
        try:
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        """Save embedding cache to file"""
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.embedding_cache, f)
    
    def get_embedding_with_cache(self, text):
        """
        Get BERT embedding for text with caching
        
        Args:
            text: Text to encode
            
        Returns:
            numpy.ndarray: Embedding vector or None if text cannot be encoded
        """
        if text is None or not text.strip():
            return None
            
        # Use a more specific cache key
        cache_key = f"{self.model_name}_{text}"
            
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Use our BertWrapper's encode method
        try:
            tokens = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if tokens.input_ids.numel() == 0:
                print(f"Warning: No valid tokens generated for text: '{text}'")
                return None
                
            with torch.no_grad():
                outputs = self.model(**tokens)
                
            last_hidden_states = outputs.last_hidden_state
            embedding = torch.mean(last_hidden_states, dim=1).numpy()
            
            if embedding is None or embedding.size == 0:
                print(f"Error: Empty embedding for text: '{text}'")
                return None
                
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            print(f"Error encoding text: '{text}': {str(e)}")
            return None
    
    def pair_and_lag_columns(self, df, columns_to_lag, suffix1='1', suffix2='2'):
        """
        Creates lagged pairs of specified columns
        
        Args:
            df: DataFrame to process
            columns_to_lag: List of column names to lag
            suffix1: Suffix for original columns
            suffix2: Suffix for lagged columns
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        for col in columns_to_lag:
            if col in df.columns:
                df[f'{col}{suffix1}'] = df[col]
                df[f'{col}{suffix2}'] = df[col].shift(-1)
        df['utter_order'] = df['participant'] + ' ' + df['participant'].shift(-1)
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
            similarities = df.apply(
                lambda row: cosine_similarity(
                    np.array(row[col1]).reshape(1, -1),
                    np.array(row[col2]).reshape(1, -1)
                )[0][0] if row[col1] is not None and row[col2] is not None else None,
                axis=1
            )
            similarity_column_name = f"{col1}_{col2}_cosine_similarity"
            df[similarity_column_name] = similarities
        return df
    
    def process_file(self, file_path):
        """
        Process a single file to compute embeddings and similarities
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            pd.DataFrame: Processed DataFrame with embeddings and similarities
        """
        # Read the file
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        
        # Pair and lag columns
        df = self.pair_and_lag_columns(df, columns_to_lag=['content'])
        
        # Compute embeddings
        for column in ["content1", "content2"]:
            df[f"{column}_embedding_{self.model_name}"] = df[column].apply(
                lambda text: self.get_embedding_with_cache(text)
            )
        
        # Calculate cosine similarities
        embedding_columns = [(f"content1_embedding_{self.model_name}", f"content2_embedding_{self.model_name}")]
        df = self.calculate_cosine_similarity(df, embedding_columns)
        
        return df
    
    def analyze_folder(self, folder_path, output_directory=None):
        """
        Analyze semantic alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Directory to save results (optional)
            
        Returns:
            pd.DataFrame: Concatenated results for all files
        """
        result_df = pd.DataFrame()
        
        # Process each file
        text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file_name in tqdm(text_files, desc=f"Processing files with {self.model_name}"):
            file_path = os.path.join(folder_path, file_name)
            df = self.process_file(file_path)
            result_df = pd.concat([result_df, df], ignore_index=True)
        
        # Save cache
        self._save_cache()
        
        # Save results if output directory is provided
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, f"semantic_alignment_{self.model_name}.csv")
            result_df.to_csv(output_path, index=False)
        
        return result_df