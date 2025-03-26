# my_package/alignment_lexsyn.py
import os
import ast
import pandas as pd
import numpy as np
from math import sqrt
from collections import Counter, OrderedDict
from nltk import ngrams
from tqdm import tqdm

class LexicalSyntacticAlignment:
    def __init__(self):
        """
        Initialize the lexical and syntactic alignment analyzer
        """
        self.model_name = "lexsyn"  # Identifier for this alignment method
    
    def safe_literal_eval(self, x):
        """
        Safely converts string representations of lists/tuples to actual Python objects
        
        Args:
            x: String representation of a list/tuple or other object
            
        Returns:
            The evaluated object or an empty list on error
        """
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            return x
        except (ValueError, SyntaxError) as e:
            return []
    
    def get_cosine_similarity(self, vec1, vec2):
        """
        Calculates cosine similarity between two vectors
        
        Args:
            vec1: First vector (as Counter or dict)
            vec2: Second vector (as Counter or dict)
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        if not vec1 or not vec2:
            return 0.0
            
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = sqrt(sum1) * sqrt(sum2)

        return float(numerator) / denominator if denominator else 0.0
    
    def compute_ngrams(self, sequence1, sequence2, ngram_size=2, ignore_duplicates=True, is_tagged=False):
        """
        Computes n-grams for two sequences
        
        Args:
            sequence1: First sequence of tokens or tagged tokens
            sequence2: Second sequence of tokens or tagged tokens
            ngram_size: Size of n-grams to compute
            ignore_duplicates: Whether to ignore duplicate word+tag n-grams between sequences
            is_tagged: Whether the sequences contain POS tags
            
        Returns:
            tuple: Four counters - lexical n-grams for each sequence and POS n-grams for each sequence
        """
        # Handle empty or None sequences
        if not sequence1 or not sequence2:
            return Counter(), Counter(), Counter(), Counter()
                
        # Generate n-grams for both sequences
        sequence1_ngrams = list(ngrams(sequence1, ngram_size))
        sequence2_ngrams = list(ngrams(sequence2, ngram_size))

        if is_tagged:
            # POS analysis
            # For tagged sequences, first check if we need to filter out exact matches
            if ignore_duplicates:
                # Remove n-grams that appear in both sequences (exact word+tag matches)
                unique_sequence1_ngrams = [ngram for ngram in sequence1_ngrams if ngram not in sequence2_ngrams]
                unique_sequence2_ngrams = [ngram for ngram in sequence2_ngrams if ngram not in sequence1_ngrams]
            else:
                # Keep all n-grams
                unique_sequence1_ngrams = sequence1_ngrams
                unique_sequence2_ngrams = sequence2_ngrams

            # Now extract only the POS tags from the n-grams (after filtering duplicates if requested)
            pos_sequence1 = [tuple(pos for _, pos in ngram) for ngram in unique_sequence1_ngrams]
            pos_sequence2 = [tuple(pos for _, pos in ngram) for ngram in unique_sequence2_ngrams]

            # Count occurrences of each POS n-gram
            pos_sequence1_count = Counter(pos_sequence1)
            pos_sequence2_count = Counter(pos_sequence2)

            # Return counts for POS analysis
            return Counter(), Counter(), pos_sequence1_count, pos_sequence2_count
        else:
            # Lexical analysis - remains unchanged
            # Join words in each n-gram into a string
            lexical_sequence1 = [' '.join(ngram) for ngram in sequence1_ngrams]
            lexical_sequence2 = [' '.join(ngram) for ngram in sequence2_ngrams]

            # Count occurrences of each lexical n-gram
            lexical_sequence1_count = Counter(lexical_sequence1)
            lexical_sequence2_count = Counter(lexical_sequence2)

            # Return counts for lexical analysis
            return lexical_sequence1_count, lexical_sequence2_count, Counter(), Counter()
    
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
        
        # Drop rows with NaN values in the lagged columns
        non_na_cols = [f"{col}{suffix2}" for col in columns_to_lag if f"{col}{suffix2}" in df.columns]
        if non_na_cols:
            return df.dropna(subset=non_na_cols)
        return df
    
    def process_file(self, file_path, max_ngram=2, lag=1, ignore_duplicates=True, add_stanford_tags=False):
        """
        Process a single file to compute lexical and syntactic alignment
        
        Args:
            file_path: Path to the file to process
            max_ngram: Maximum n-gram size to compute (default: 2)
            lag: Number of turns to lag when pairing utterances (default: 1)
            ignore_duplicates: Whether to ignore duplicate n-grams (default: True)
            add_stanford_tags: Whether to include Stanford POS tags (default: False)
            
        Returns:
            pd.DataFrame: Processed DataFrame with alignment metrics
        """
        try:
            # Read the file
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            
            # Skip if file is empty or has just one row
            if df.empty or len(df) <= 1:
                print(f"Skipping invalid file: {file_path}")
                return pd.DataFrame()
            
            # Convert string representations of lists/tuples to actual Python objects
            for col in ['token', 'lemma', 'tagged_token', 'tagged_lemma', 'tagged_stan_token', 'tagged_stan_lemma']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: self.safe_literal_eval(x))
            
            # Pair and lag columns
            tag_columns = ['tagged_token', 'tagged_lemma']
            if add_stanford_tags:
                tag_columns.extend(['tagged_stan_token', 'tagged_stan_lemma'])
                
            columns_to_lag = ['token', 'lemma', 'content'] + tag_columns
            df = self.pair_and_lag_columns(df, columns_to_lag, lag=lag)
            
            # For empty dataframe after lag, return empty result
            if df.empty:
                return pd.DataFrame()
            
            # Save original filename for reference
            df['source_file'] = os.path.basename(file_path)
            
            # Add lag information to the dataframe
            df['lag'] = lag
            
            # Initialize time counter
            time_counter = 1
            
            # Calculate n-grams and cosine similarity
            alignment_data = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
                results = OrderedDict()
                
                # Add time and source info
                results['time'] = time_counter
                time_counter += 1
                results['source_file'] = os.path.basename(file_path)
                results['lag'] = lag
                
                # Add participant if available
                if 'participant' in row:
                    results['participant'] = row['participant']
                
                # Add original content and token data if available
                for col in ['content', 'token', 'lemma', 'tagged_token', 'tagged_lemma']:
                    if col in row:
                        results[col] = row[col]
                
                # Add Stanford tags if available and requested
                if add_stanford_tags:
                    for col in ['tagged_stan_token', 'tagged_stan_lemma']:
                        if col in row:
                            results[col] = row[col]
                
                # Add utterance order if available
                if 'utter_order' in row:
                    results['utter_order'] = row['utter_order']
                
                # Add lagged content
                for col in ['content1', 'content2']:
                    if col in row:
                        results[col] = row[col]
                
                # Add utterance lengths
                results['utterance_length1'] = len(row['token1']) if isinstance(row.get('token1'), list) else 0
                results['utterance_length2'] = len(row['token2']) if isinstance(row.get('token2'), list) else 0
                
                # Calculate n-gram similarities for multiple n-gram sizes
                for n in range(1, max_ngram + 1):
                    # Process if token columns exist
                    if all(col in row for col in ['token1', 'token2']):
                        # Lexical analysis for tokens
                        lex_tok1_count, lex_tok2_count, _, _ = self.compute_ngrams(
                            row['token1'], row['token2'], ngram_size=n, ignore_duplicates=False, is_tagged=False)
                        results[f'lexical_tok{n}_cosine'] = self.get_cosine_similarity(lex_tok1_count, lex_tok2_count)
                    
                    # Process if lemma columns exist
                    if all(col in row for col in ['lemma1', 'lemma2']):
                        # Lexical analysis for lemmas
                        lex_lem1_count, lex_lem2_count, _, _ = self.compute_ngrams(
                            row['lemma1'], row['lemma2'], ngram_size=n, ignore_duplicates=False, is_tagged=False)
                        results[f'lexical_lem{n}_cosine'] = self.get_cosine_similarity(lex_lem1_count, lex_lem2_count)
                    
                    # Process if tagged token columns exist
                    if all(col in row for col in ['tagged_token1', 'tagged_token2']):
                        # POS analysis for tokens
                        _, _, pos_tok1_count, pos_tok2_count = self.compute_ngrams(
                            row['tagged_token1'], row['tagged_token2'], 
                            ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
                        results[f'pos_tok{n}_cosine'] = self.get_cosine_similarity(pos_tok1_count, pos_tok2_count)
                    
                    # Process if tagged lemma columns exist
                    if all(col in row for col in ['tagged_lemma1', 'tagged_lemma2']):
                        # POS analysis for lemmas
                        _, _, pos_lem1_count, pos_lem2_count = self.compute_ngrams(
                            row['tagged_lemma1'], row['tagged_lemma2'], 
                            ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
                        results[f'pos_lem{n}_cosine'] = self.get_cosine_similarity(pos_lem1_count, pos_lem2_count)
                    
                    # Optional Stanford POS tags
                    if add_stanford_tags:
                        if all(col in row for col in ['tagged_stan_token1', 'tagged_stan_token2']):
                            # Stanford POS analysis for tokens
                            _, _, stan_tok1_count, stan_tok2_count = self.compute_ngrams(
                                row['tagged_stan_token1'], row['tagged_stan_token2'], 
                                ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
                            results[f'stan_pos_tok{n}_cosine'] = self.get_cosine_similarity(stan_tok1_count, stan_tok2_count)
                        
                        if all(col in row for col in ['tagged_stan_lemma1', 'tagged_stan_lemma2']):
                            # Stanford POS analysis for lemmas
                            _, _, stan_lem1_count, stan_lem2_count = self.compute_ngrams(
                                row['tagged_stan_lemma1'], row['tagged_stan_lemma2'], 
                                ngram_size=n, ignore_duplicates=ignore_duplicates, is_tagged=True)
                            results[f'stan_pos_lem{n}_cosine'] = self.get_cosine_similarity(stan_lem1_count, stan_lem2_count)
                
                # Calculate composite scores
                # Lexical gets all n-grams (n=1 and above)
                lexical_cols = [col for col in results.keys() if col.startswith('lexical_') and col.endswith('_cosine')]
                
                # Syntactic POS only uses n-grams of length 2 and above
                pos_cols = [col for col in results.keys() if (col.startswith('pos_') or col.startswith('stan_pos_')) 
                        and col.endswith('_cosine') and not (col.endswith('1_cosine'))]
                
                if lexical_cols:
                    results['lexical_master_cosine'] = np.mean([results[col] for col in lexical_cols])
                
                if pos_cols:
                    results['syntactic_master_cosine'] = np.mean([results[col] for col in pos_cols])
                
                alignment_data.append(results)
            
            # Convert results to DataFrame
            df_results = pd.DataFrame(alignment_data)
            
            if df_results.empty:
                return pd.DataFrame()
            
            # Define column order based on specifications
            column_order = [
                'time', 'source_file', 'participant', 'content', 'token', 'lemma', 
                'tagged_token', 'tagged_lemma'
            ]
            
            # Add Stanford tags if available
            if add_stanford_tags:
                column_order.extend(['tagged_stan_token', 'tagged_stan_lemma'])
            
            # Continue with the rest of the columns
            column_order.extend([
                'lag', 'utter_order', 'content1', 'content2', 
                'utterance_length1', 'utterance_length2'
            ])
            
            # Add n-gram metrics in order
            for n in range(1, max_ngram + 1):
                # Lexical metrics
                column_order.extend([
                    f'lexical_tok{n}_cosine', f'lexical_lem{n}_cosine'
                ])
                
                # POS metrics
                column_order.extend([
                    f'pos_tok{n}_cosine', f'pos_lem{n}_cosine'
                ])
                
                # Stanford POS metrics if available
                if add_stanford_tags:
                    column_order.extend([
                        f'stan_pos_tok{n}_cosine', f'stan_pos_lem{n}_cosine'
                    ])
            
            # Add master metrics at the end
            column_order.extend(['lexical_master_cosine', 'syntactic_master_cosine'])
            
            # Filter to only include columns that exist in the dataframe
            final_column_order = [col for col in column_order if col in df_results.columns]
            
            # Add any remaining columns not in our predefined order
            for col in df_results.columns:
                if col not in final_column_order:
                    final_column_order.append(col)
            
            # Reorder the dataframe
            df_results = df_results[final_column_order]
            
            return df_results
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for debugging
            return pd.DataFrame()  # Return empty dataframe on error

    def analyze_folder(self, folder_path, output_directory=None, file_pattern="*.txt", lag=1, 
                    max_ngram=2, ignore_duplicates=True, add_stanford_tags=False, **kwargs):
        """
        Analyze lexical and syntactic alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Directory to save results (optional)
            file_pattern: Pattern to match text files (default: "*.txt")
            lag: Number of turns to lag when pairing utterances (default: 1)
            max_ngram: Maximum n-gram size to compute (default: 2)
            ignore_duplicates: Whether to ignore duplicate n-grams (default: True)
            add_stanford_tags: Whether to include Stanford POS tags (default: False)
            **kwargs: Additional arguments passed to the underlying analyzer
            
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
                # Pass all parameters to process_file
                file_df = self.process_file(
                    file_path=file_path, 
                    max_ngram=max_ngram,
                    lag=lag,
                    ignore_duplicates=ignore_duplicates,
                    add_stanford_tags=add_stanford_tags
                )
                
                if not file_df.empty:
                    result_df = pd.concat([result_df, file_df], ignore_index=True)
                    successful_files += 1
            except Exception as e:
                print(f"Fatal error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Successfully processed {successful_files} out of {len(file_paths)} files")
        
        # Save results if output directory is provided
        if output_directory and not result_df.empty:
            os.makedirs(output_directory, exist_ok=True)
            
            # Create a filename that includes all parameters
            dup_str = "noDups" if ignore_duplicates else "withDups"
            stan_str = "withStan" if add_stanford_tags else "noStan"
            
            output_path = os.path.join(
                output_directory, 
                f"lexsyn_alignment_ngram{max_ngram}_lag{lag}_{dup_str}_{stan_str}.csv"
            )
            
            result_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return result_df




