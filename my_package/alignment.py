# my_package/alignment.py
import os
from .alignment_bert import SemanticAlignmentAnalyzer
from .alignment_fasttext import SemanticAlignmentFastText
from .alignment_lexsyn import LexicalSyntacticAlignment
from .surrogates import SurrogateAlignment, SurrogateGenerator

class LinguisticAlignment:
    def __init__(self, alignment_type=None, alignment_types=None, **kwargs):
        """
        Initialize with one or more linguistic alignment analyzers
        
        Args:
            alignment_type: String specifying analyzer type (backward compatibility)
            alignment_types: String or list of strings specifying analyzer types
            **kwargs: Configuration parameters (model_name, cache_dir, etc.)
        """
        # Handle both parameters for backward compatibility
        if alignment_type and not alignment_types:
            alignment_types = [alignment_type]
        elif alignment_types is None:
            alignment_types = ["bert"]  # Default
        elif alignment_types == "all":
            alignment_types = ["bert", "fasttext", "lexsyn"]
        elif isinstance(alignment_types, str):
            alignment_types = [alignment_types]
            
        self.analyzers = {}
        self.cache_dir = kwargs.get("cache_dir")
        
        for align_type in alignment_types:
            align_type = align_type.lower()
            
            if align_type == "bert":
                model_name = kwargs.get("model_name", "bert-base-uncased")
                token = kwargs.get("token")
                self.analyzers["bert"] = SemanticAlignmentAnalyzer(
                    model_name=model_name, 
                    token=token
                )
                
            elif align_type == "fasttext":
                model_name = kwargs.get("model_name", "fasttext-wiki-news-300")
                self.analyzers["fasttext"] = SemanticAlignmentFastText(
                    model_name=model_name, 
                    cache_dir=self.cache_dir
                )
                
            elif align_type == "lexsyn":
                self.analyzers["lexsyn"] = LexicalSyntacticAlignment()
            else:
                raise ValueError(f"Unsupported alignment type: {align_type}")
        
        # Store the primary alignment type for backward compatibility
        self.alignment_type = alignment_types[0] if alignment_types else None
    
    def analyze_folder(self, folder_path, output_directory=None, **kwargs):
        """
        Analyze alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Root directory to save results
            **kwargs: Parameters for various analyzers
            
        Returns:
            pd.DataFrame: Concatenated results or merged results from multiple analyzers
        """
        results = {}
        
        for analyzer_type, analyzer in self.analyzers.items():
            # Create analyzer-specific output directory
            analyzer_dir = None
            if output_directory:
                analyzer_dir = os.path.join(output_directory, analyzer_type)
                os.makedirs(analyzer_dir, exist_ok=True)
            
            # Filter kwargs to only include relevant parameters for this analyzer
            filtered_kwargs = self._filter_kwargs_for_analyzer(analyzer_type, kwargs)
            
            # Run analysis with filtered parameters
            analyzer_results = analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=analyzer_dir,
                **filtered_kwargs
            )
            
            results[analyzer_type] = analyzer_results
        
        # If only one analyzer, return its results directly
        if len(results) == 1:
            return next(iter(results.values()))
        
        # Otherwise merge results from multiple analyzers
        return self._merge_results(results)
    
    def process_file(self, file_path, lag=1, high_sd_cutoff=3, low_n_cutoff=1, max_ngram=2,
                    ignore_duplicates=True, add_stanford_tags=False, **kwargs):
        """
        Process a single file to compute alignment metrics
        
        Args:
            file_path: Path to the file to process
            lag: Number of turns to lag when pairing utterances (default: 1)
            high_sd_cutoff: Standard deviation cutoff for high-frequency words (FastText only)
            low_n_cutoff: Minimum frequency cutoff (FastText only)
            max_ngram: Maximum n-gram size to compute (LexSyn only)
            ignore_duplicates: Whether to ignore duplicate n-grams (LexSyn only)
            add_stanford_tags: Whether to include Stanford POS tags (LexSyn only)
            **kwargs: Additional arguments passed to the underlying analyzer
            
        Returns:
            pd.DataFrame: Processed DataFrame with alignment metrics
        """
        if self.alignment_type == "bert":
            # BERT implementation
            return self.analyzer.process_file(file_path=file_path, lag=lag, **kwargs)
        elif self.alignment_type == "fasttext":
            # FastText implementation
            return self.analyzer.process_file(
                file_path=file_path, 
                lag=lag, 
                high_sd_cutoff=high_sd_cutoff,
                low_n_cutoff=low_n_cutoff,
                **kwargs
            )
        elif self.alignment_type == "lexsyn":
            # LexSyn implementation
            return self.analyzer.process_file(
                file_path=file_path,
                lag=lag,
                max_ngram=max_ngram,
                ignore_duplicates=ignore_duplicates,
                add_stanford_tags=add_stanford_tags,
                **kwargs
            )
    
    def analyze_baseline(self, input_files, output_directory=None, surrogate_directory=None,
                        use_existing_surrogates=None, **kwargs):
        """
        Generate surrogate conversation pairs and analyze their alignment
        
        Args:
            input_files: Path to directory containing conversation files or list of files
            output_directory: Root directory to save results
            surrogate_directory: Directory to save new surrogate files
            use_existing_surrogates: Path to existing surrogate files to use
            **kwargs: Additional parameters
        """
        results = {}
        
        for analyzer_type, analyzer in self.analyzers.items():
            # Create analyzer-specific output directory
            analyzer_dir = None
            if output_directory:
                analyzer_dir = os.path.join(output_directory, analyzer_type)
                os.makedirs(analyzer_dir, exist_ok=True)
            
            # Create surrogate analyzer with same parameters
            surrogate_analyzer = SurrogateAlignment(
                alignment_type=analyzer_type,
                cache_dir=self.cache_dir
            )
            
            # Filter kwargs to only include relevant parameters
            filtered_kwargs = self._filter_kwargs_for_analyzer(analyzer_type, kwargs)
            
            # Run baseline analysis
            baseline_results = surrogate_analyzer.analyze_baseline(
                input_files=input_files,
                output_directory=analyzer_dir,
                surrogate_directory=surrogate_directory,
                use_existing_surrogates=use_existing_surrogates,
                **filtered_kwargs
            )
            
            results[analyzer_type] = baseline_results
        
        # If only one analyzer, return its results directly
        if len(results) == 1:
            return next(iter(results.values()))
        
        # Otherwise merge results from multiple analyzers
        return self._merge_results(results)
    
    def _filter_kwargs_for_analyzer(self, analyzer_type, kwargs):
        """
        Filter kwargs to only include relevant parameters for this analyzer
        
        Args:
            analyzer_type: Type of analyzer ("bert", "fasttext", "lexsyn")
            kwargs: Dictionary of all parameters
            
        Returns:
            dict: Filtered parameters relevant to the specified analyzer
        """
        # Define which parameters apply to which analyzers
        common_params = ["lag", "file_pattern", "output_directory"]
        
        bert_params = ["model_name", "token"]
        
        fasttext_params = ["model_name", "high_sd_cutoff", "low_n_cutoff", 
                        "save_vocab"]
        
        lexsyn_params = ["max_ngram", "ignore_duplicates", "add_stanford_tags"]
        
        surrogate_params = ["all_surrogates", "keep_original_turn_order", 
                        "id_separator", "condition_label", "dyad_label"]
        
        # Determine which parameters to include
        if analyzer_type == "bert":
            valid_params = common_params + bert_params + surrogate_params
        elif analyzer_type == "fasttext":
            valid_params = common_params + fasttext_params + surrogate_params
        elif analyzer_type == "lexsyn":
            valid_params = common_params + lexsyn_params + surrogate_params
        else:
            valid_params = common_params + surrogate_params
        
        # Filter kwargs to only include valid parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return filtered_kwargs

    def _merge_results(self, results_dict):
        """
        Merge results from multiple analyzers into a single DataFrame
        
        Args:
            results_dict: Dictionary mapping analyzer type to DataFrame
            
        Returns:
            pd.DataFrame: Merged results
        """
        if not results_dict:
            return pd.DataFrame()
        
        # Start with the first dataframe
        analyzer_types = list(results_dict.keys())
        base_df = results_dict[analyzer_types[0]].copy()
        
        # Identify common columns (metadata) that exist in all dataframes
        common_cols = set(base_df.columns)
        for df in results_dict.values():
            common_cols &= set(df.columns)
        
        # Remove embedding columns from common columns
        common_cols = [col for col in common_cols if 'embedding' not in col and '_dims' not in col]
        
        # For each additional analyzer, add its unique columns to the base DataFrame
        for analyzer_type in analyzer_types[1:]:
            df = results_dict[analyzer_type]
            
            # Find unique columns in this dataframe (excluding common columns)
            unique_cols = [col for col in df.columns if col not in common_cols or col.endswith('_cosine_similarity')]
            
            # Add unique columns to base dataframe
            for col in unique_cols:
                # Rename similarity columns to include analyzer type
                if col.endswith('_cosine_similarity'):
                    new_col = f"{analyzer_type}_{col}"
                    base_df[new_col] = df[col]
                # Add other unique columns directly
                elif col not in base_df.columns:
                    base_df[col] = df[col]
        
        return base_df