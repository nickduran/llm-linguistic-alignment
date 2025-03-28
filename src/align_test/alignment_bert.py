# my_package/alignment_bert.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from .bert_model import BertWrapper

class SemanticAlignmentAnalyzer:
    def __init__(self, model_name="bert-base-uncased", token=None):
        """
        Initialize the semantic alignment analyzer
        
        Args:
            model_name: Name of the BERT model to use
            token: Hugging Face token (optional)
        """
        self.bert_wrapper = BertWrapper(model_name, token)
        self.tokenizer = self.bert_wrapper.tokenizer
        self.model = self.bert_wrapper.model
        self.model_name = model_name.split('/')[-1]  # Extract model name for column naming
    
    def get_embedding(self, text):
        """
        Get BERT embedding for text
        
        Args:
            text: Text to encode
            
        Returns:
            numpy.ndarray: Embedding vector or None if text cannot be encoded
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return None
        
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
            embedding = torch.mean(last_hidden_states, dim=1).detach().numpy().squeeze()
            
            if embedding.size == 0:
                print(f"Error: Empty embedding for text: '{text}'")
                return None
                
            return embedding
            
        except Exception as e:
            print(f"Error encoding text: '{text}': {str(e)}")
            return None
    
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
                if isinstance(row.get(col1), list) and isinstance(row.get(col2), list):
                    # Convert lists back to numpy arrays for similarity calculation
                    embed1 = np.array(row[col1])
                    embed2 = np.array(row[col2])
                    
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
            
            # Pair and lag columns - ensure participant exists
            if 'participant' in df.columns:
                df = self.pair_and_lag_columns(df, columns_to_lag=['content'], lag=lag)
            else:
                print(f"Warning: 'participant' column not found in {file_path}, skipping participant tracking")
                # Create lagged columns without participant tracking
                for col in ['content']:
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
            
            for column in ["content1", "content2"]:
                if column in df.columns:
                    col_name = f"{column}_embedding_{self.model_name}"
                    df[col_name] = None  # Initialize with None
                    
                    # Process row by row
                    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {column}"):
                        if pd.notna(row[column]) and isinstance(row[column], str):
                            # Get the embedding
                            embedding = self.get_embedding(row[column])
                            
                            # Store the embedding directly in the DataFrame
                            if embedding is not None:
                                # Store the embedding as a serialized representation
                                # This makes it viewable and preserves the data
                                df.at[idx, col_name] = embedding.tolist()
                                
                                # Add a dimension indicator column for verification
                                df.at[idx, f"{col_name}_dims"] = embedding.shape[0] if hasattr(embedding, 'shape') else 0
                    
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
    
    def analyze_folder(self, folder_path, output_directory=None, file_pattern="*.txt", lag=1):
        """
        Analyze semantic alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Directory to save results (optional)
            file_pattern: Pattern to match text files (default: "*.txt")
            lag: Number of turns to lag when pairing utterances (default: 1)
            
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
        
        # Rename the similarity column to a simpler format
        old_col_name = f"content1_embedding_{self.model_name}_content2_embedding_{self.model_name}_cosine_similarity"
        new_col_name = f"{self.model_name}_cosine_similarity"
        
        if old_col_name in result_df.columns:
            result_df = result_df.rename(columns={old_col_name: new_col_name})
        
        # Reorder and select only the specified columns
        desired_columns = [
            "source_file",
            "participant",
            "content",
            "lag",
            "content1",
            "content2",
            "utter_order",
            f"content1_embedding_{self.model_name}",
            f"content1_embedding_{self.model_name}_dims",
            f"content2_embedding_{self.model_name}",
            f"content2_embedding_{self.model_name}_dims",
            new_col_name
        ]
        
        # Filter to include only columns that exist in the DataFrame
        final_columns = [col for col in desired_columns if col in result_df.columns]
        
        # Reorder the columns
        result_df = result_df[final_columns]
        
        # Save results if output directory is provided
        if output_directory and not result_df.empty:
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, f"semantic_alignment_{self.model_name}_lag{lag}.csv")
            result_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return result_df