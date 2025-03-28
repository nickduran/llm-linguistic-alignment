# my_package/alignment.py
import os
import glob
import pandas as pd
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
    
    def analyze_folder(self, folder_path, output_directory=None, file_pattern="*.txt", lag=1, **kwargs):
        """
        Analyze alignment for all text files in a folder
        """
        print(f"ANALYZE_FOLDER: Processing data from folder: {folder_path} with lag={lag}")
        results = {}
        
        # Ensure lag parameter is explicitly set in kwargs
        kwargs['lag'] = lag
        
        for analyzer_type, analyzer in self.analyzers.items():
            # Create analyzer-specific output directory
            analyzer_dir = None
            if output_directory:
                analyzer_dir = os.path.join(output_directory, analyzer_type)
                os.makedirs(analyzer_dir, exist_ok=True)
            
            # Filter kwargs to only include relevant parameters for this analyzer
            filtered_kwargs = self._filter_kwargs_for_analyzer(analyzer_type, kwargs)
            
            # Run analysis with filtered parameters 
            # Important: Don't provide output_directory here to prevent duplicate files
            analyzer_results = analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=None,  # Don't save intermediate results
                file_pattern=file_pattern,
                **filtered_kwargs
            )
            
            # Store results
            results[analyzer_type] = analyzer_results
            
            # If output directory is provided, save cleaned results (without embeddings)
            if analyzer_dir and not analyzer_results.empty:
                # Remove embedding columns and dimension columns
                cols_to_remove = [col for col in analyzer_results.columns 
                                if 'embedding' in col or '_dims' in col]
                
                clean_results = analyzer_results.drop(columns=cols_to_remove, errors='ignore')
                
                # Create the appropriate filename based on analyzer type
                # Use the actual lag value from filtered_kwargs
                current_lag = filtered_kwargs.get('lag', 1)
                
                if analyzer_type == "lexsyn":
                    max_ngram = filtered_kwargs.get('max_ngram', 2)
                    ignore_duplicates = filtered_kwargs.get('ignore_duplicates', True)
                    add_stanford_tags = filtered_kwargs.get('add_stanford_tags', False)
                    dup_str = "noDups" if ignore_duplicates else "withDups"
                    stan_str = "withStan" if add_stanford_tags else "noStan"
                    output_path = os.path.join(
                        analyzer_dir, 
                        f"lexsyn_alignment_ngram{max_ngram}_lag{current_lag}_{dup_str}_{stan_str}.csv"
                    )
                elif analyzer_type == "fasttext":
                    high_sd_cutoff = filtered_kwargs.get('high_sd_cutoff', 3)
                    low_n_cutoff = filtered_kwargs.get('low_n_cutoff', 1)
                    sd_str = f"sd{high_sd_cutoff}"
                    n_str = f"n{low_n_cutoff}"
                    output_path = os.path.join(
                        analyzer_dir, 
                        f"semantic_alignment_{analyzer_type}_lag{current_lag}_{sd_str}_{n_str}.csv"
                    )
                elif analyzer_type == "bert":
                    model_name = getattr(analyzer, 'model_name', 'bert-base-uncased')
                    output_path = os.path.join(
                        analyzer_dir, 
                        f"semantic_alignment_{model_name}_lag{current_lag}.csv"
                    )
                else:
                    output_path = os.path.join(
                        analyzer_dir, 
                        f"alignment_{analyzer_type}_lag{current_lag}.csv"
                    )
                
                # Save the clean results
                clean_results.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
        
        # If only one analyzer, return its results directly
        if len(results) == 1:
            return next(iter(results.values()))
        
        # Otherwise merge results from multiple analyzers and save to output directory
        return self._merge_results(results, output_directory)
    
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
    
    def analyze_baseline(self, input_files, output_directory="results", surrogate_directory=None,
                        use_existing_surrogates=None, **kwargs):
        """
        Generate surrogate conversation pairs and analyze their alignment as a baseline
        
        Args:
            input_files: Path to directory containing conversation files or list of file paths
            output_directory: Directory to save alignment results (default: "results")
            surrogate_directory: Directory to save surrogate files (optional)
            use_existing_surrogates: Path to existing surrogate files to use instead of generating new ones
            **kwargs: Additional arguments for alignment analysis
            
        Returns:
            pd.DataFrame: Baseline alignment results for surrogate pairs
        """
        # Add these debug lines
        print(f"ANALYZE_BASELINE: Input files from: {input_files}")
        print(f"ANALYZE_BASELINE: Output directory: {output_directory}")

        results = {}
        
        # Handle surrogate generation/loading at the higher level
        surrogate_files = None
        
        if use_existing_surrogates is None and surrogate_directory is None:
            # Default behavior: create surrogates in a folder inside output_directory
            surrogate_directory = os.path.join(output_directory, "surrogates")
        
        if use_existing_surrogates is not None:
            # Use existing surrogates
            if not os.path.isdir(use_existing_surrogates):
                raise ValueError(f"Specified surrogate directory does not exist: {use_existing_surrogates}")
                
            surrogate_files = glob.glob(os.path.join(use_existing_surrogates, "*.txt"))
            if not surrogate_files:
                raise ValueError(f"No surrogate files found in {use_existing_surrogates}")
                
            print(f"Using {len(surrogate_files)} existing surrogate files from {use_existing_surrogates}")
        else:
            # Generate new surrogates once for all analyzers
            os.makedirs(surrogate_directory, exist_ok=True)
            
            # Extract file format parameters
            id_separator = kwargs.get('id_separator', '-')
            condition_label = kwargs.get('condition_label', 'cond')
            dyad_label = kwargs.get('dyad_label', 'dyad')
            all_surrogates = kwargs.get('all_surrogates', True)
            keep_original_turn_order = kwargs.get('keep_original_turn_order', True)
            
            # Resolve input files
            if isinstance(input_files, str) and os.path.isdir(input_files):
                file_list = glob.glob(os.path.join(input_files, "*.txt"))
            elif isinstance(input_files, list):
                file_list = input_files
            else:
                raise ValueError("input_files must be a directory path or list of file paths")
            
            # Generate surrogate conversation pairs
            print(f"Generating surrogate conversation pairs from {len(file_list)} files...")
            surrogate_generator = SurrogateGenerator()
            surrogate_generator.output_directory = surrogate_directory
            surrogate_files = surrogate_generator.generate_surrogates(
                original_conversation_list=file_list,
                all_surrogates=all_surrogates,
                keep_original_turn_order=keep_original_turn_order,
                id_separator=id_separator,
                dyad_label=dyad_label,
                condition_label=condition_label
            )
            
            if not surrogate_files:
                raise ValueError("No surrogate files were generated. Please check your input files and parameters.")
                
            print(f"Generated {len(surrogate_files)} surrogate files in {surrogate_directory}")
        
        # Now analyze these surrogates with each analyzer
        for analyzer_type, analyzer in self.analyzers.items():
            # Get model name and token from analyzer if available
            model_name = None
            token = None
            
            if analyzer_type == "bert" and hasattr(analyzer, "bert_wrapper"):
                model_name = analyzer.model_name
                token = analyzer.bert_wrapper.token
            elif analyzer_type == "fasttext" and hasattr(analyzer, "fasttext_wrapper"):
                model_name = analyzer.model_name
            
            # Create analyzer-specific output directory
            analyzer_output_dir = None
            if output_directory:
                analyzer_output_dir = os.path.join(output_directory, analyzer_type)
                os.makedirs(analyzer_output_dir, exist_ok=True)
            
            # Filter kwargs to only include relevant parameters
            filtered_kwargs = self._filter_kwargs_for_analyzer(analyzer_type, kwargs)
            
            # Debug output to troubleshoot path issues
            surrogate_folder = os.path.dirname(surrogate_files[0])
            print(f"DEBUG: Surrogate folder path: {surrogate_folder}")
            print(f"DEBUG: Number of files in surrogate folder: {len(glob.glob(os.path.join(surrogate_folder, '*.txt')))}")
            print(f"DEBUG: First few surrogate files: {glob.glob(os.path.join(surrogate_folder, '*.txt'))[:3]}")
            
            # Analyze surrogate files directly, bypassing the surrogate generation
            print(f"Analyzing surrogate files with {analyzer_type}...")
            
            # Before calling analyze_folder, make sure surrogate parameters are removed
            analysis_params = {k: v for k, v in filtered_kwargs.items() 
                            if k not in ["all_surrogates", "keep_original_turn_order", 
                                        "id_separator", "condition_label", "dyad_label",
                                        "use_existing_surrogates", "surrogate_directory"]}
            
            original_output_dir = analyzer_output_dir
            analysis_results = analyzer.analyze_folder(
                folder_path=surrogate_folder,
                file_pattern="*.txt",
                output_directory=None,  # Don't save intermediate results
                **analysis_params
            )
            
            # Save results with baseline-specific filename in analyzer directory
            if not analysis_results.empty:
                # Remove embedding columns
                cols_to_remove = [col for col in analysis_results.columns if 'embedding' in col or '_dims' in col]
                clean_results = analysis_results.drop(columns=cols_to_remove, errors='ignore')
                
                # Create baseline-specific filename with parameter details
                baseline_path = os.path.join(analyzer_output_dir, f"baseline_alignment_{analyzer_type}_lag{analysis_params.get('lag', 1)}.csv")
                
                # Add model-specific parameters to filename
                if analyzer_type == "lexsyn":
                    max_ngram = analysis_params.get('max_ngram', 2)
                    ignore_duplicates = analysis_params.get('ignore_duplicates', True)
                    add_stanford_tags = analysis_params.get('add_stanford_tags', False)
                    dup_str = "noDups" if ignore_duplicates else "withDups"
                    stan_str = "withStan" if add_stanford_tags else "noStan"
                    baseline_path = os.path.join(
                        analyzer_output_dir, 
                        f"baseline_alignment_lexsyn_ngram{max_ngram}_lag{analysis_params.get('lag', 1)}_{dup_str}_{stan_str}.csv"
                    )
                elif analyzer_type == "fasttext":
                    high_sd_cutoff = analysis_params.get('high_sd_cutoff', 3)
                    low_n_cutoff = analysis_params.get('low_n_cutoff', 1)
                    sd_str = f"sd{high_sd_cutoff}"
                    n_str = f"n{low_n_cutoff}"
                    baseline_path = os.path.join(
                        analyzer_output_dir, 
                        f"baseline_alignment_fasttext_lag{analysis_params.get('lag', 1)}_{sd_str}_{n_str}.csv"
                    )
                
                elif analyzer_type == "bert":
                    # Get model name (either from analyzer or use a default)
                    model_name = model_name or "bert-base-uncased"
                    baseline_path = os.path.join(
                        analyzer_output_dir, 
                        f"baseline_alignment_{model_name}_lag{analysis_params.get('lag', 1)}.csv"
                    )

                # Save the results
                clean_results.to_csv(baseline_path, index=False)
                print(f"Baseline alignment results saved to {baseline_path}")
            
            results[analyzer_type] = analysis_results
        
        # Return results
        if len(results) == 1:
            return next(iter(results.values()))
        else:
            return self._merge_results(results)
    
    def _filter_kwargs_for_analyzer(self, analyzer_type, kwargs):
        """Filter kwargs to only include relevant parameters for this analyzer"""
        
        # Define which parameters apply to which analyzers
        common_params = ["lag", "file_pattern"]
        
        bert_params = ["model_name", "token"]
        
        fasttext_params = ["model_name", "high_sd_cutoff", "low_n_cutoff", 
                        "save_vocab"]
        
        lexsyn_params = ["max_ngram", "ignore_duplicates", "add_stanford_tags"]
        
        # Define surrogate-specific parameters that should be excluded from analyzer calls
        surrogate_params = ["all_surrogates", "keep_original_turn_order", 
                        "id_separator", "condition_label", "dyad_label", 
                        "use_existing_surrogates", "surrogate_directory"]
        
        # Determine which parameters to include
        if analyzer_type == "bert":
            valid_params = common_params + bert_params
        elif analyzer_type == "fasttext":
            valid_params = common_params + fasttext_params
        elif analyzer_type == "lexsyn":
            valid_params = common_params + lexsyn_params
        else:
            valid_params = common_params
        
        # Filter kwargs to only include valid parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and k not in surrogate_params}
        
        return filtered_kwargs

    def _merge_results(self, results_dict, output_directory=None):
        """
        Merge results from multiple analyzers into a single DataFrame
        
        Args:
            results_dict: Dictionary mapping analyzer type to DataFrame
            output_directory: Directory to save the merged results (optional)
            
        Returns:
            pd.DataFrame: Merged results
        """
        if not results_dict:
            return pd.DataFrame()
        
        # Start with the first dataframe
        analyzer_types = list(results_dict.keys())
        base_df = results_dict[analyzer_types[0]].copy()
        
        # Check for row count differences and warn if they exist
        row_counts = {analyzer: len(df) for analyzer, df in results_dict.items()}
        if len(set(row_counts.values())) > 1:
            print("WARNING: Different analyzers returned different numbers of rows:")
            for analyzer, count in row_counts.items():
                print(f"  {analyzer}: {count} rows")
            print("This might be due to different analyzers skipping certain rows based on their specific requirements.")
            print("(e.g., if a row lacks necessary data for that analyzer)")
        
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
                # Skip embedding and dimension columns
                if 'embedding' in col or '_dims' in col:
                    continue
                    
                # Rename similarity columns to include analyzer type
                if col.endswith('_cosine_similarity'):
                    new_col = f"{analyzer_type}_{col}"
                    base_df[new_col] = df[col]
                # Add other unique columns directly
                elif col not in base_df.columns:
                    base_df[col] = df[col]
        
        # Ensure no embedding or dimension columns are in the final result
        cols_to_remove = [col for col in base_df.columns if 'embedding' in col or '_dims' in col]
        if cols_to_remove:
            base_df = base_df.drop(columns=cols_to_remove, errors='ignore')

        # Save merged results if output directory is provided
        if output_directory and not base_df.empty:
            os.makedirs(output_directory, exist_ok=True)
            merged_output_path = os.path.join(output_directory, "merged_alignment_results.csv")
            base_df.to_csv(merged_output_path, index=False)
            print(f"Merged results from {', '.join(analyzer_types)} saved to {merged_output_path}")
        
        return base_df