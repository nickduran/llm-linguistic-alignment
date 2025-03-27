# my_package/alignment.py
import os
from .alignment_bert import SemanticAlignmentAnalyzer
from .alignment_w2v import SemanticAlignmentW2V
from .alignment_lexsyn import LexicalSyntacticAlignment
from .surrogates import SurrogateAlignment, SurrogateGenerator

class SemanticAlignment:
    def __init__(self, embedding_model="bert", model_name=None, token=None, cache_dir=None):
        """
        Initialize a semantic alignment analyzer with a specified embedding model
        
        Args:
            embedding_model: Type of embedding model to use ('bert', 'word2vec', or 'lexsyn')
            model_name: Name of the specific model to use (optional)
            token: API token for model access (optional, needed for BERT only)
            cache_dir: Directory to cache models (optional)
        """
        self.embedding_model = embedding_model.lower()
        
        if self.embedding_model == "bert":
            model_name = model_name or "bert-base-uncased"
            self.analyzer = SemanticAlignmentAnalyzer(model_name=model_name, token=token)
        elif self.embedding_model == "word2vec":
            model_name = model_name or "word2vec-google-news-300"
            self.analyzer = SemanticAlignmentW2V(model_name=model_name, cache_dir=cache_dir)
        elif self.embedding_model == "lexsyn":
            self.analyzer = LexicalSyntacticAlignment()
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}. Use 'bert', 'word2vec', or 'lexsyn'.")
    
    def analyze_folder(self, folder_path, output_directory=None, file_pattern="*.txt", lag=1, 
                    high_sd_cutoff=3, low_n_cutoff=1, save_vocab=True, max_ngram=2, 
                    ignore_duplicates=True, add_stanford_tags=False, **kwargs):
        """
        Analyze alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Root directory to save results (optional)
            file_pattern: Pattern to match text files (default: "*.txt")
            lag: Number of turns to lag when pairing utterances (default: 1)
            high_sd_cutoff: Standard deviation cutoff for high-frequency words (Word2Vec only)
            low_n_cutoff: Minimum frequency cutoff (Word2Vec only)
            save_vocab: Whether to save vocabulary lists (Word2Vec only)
            max_ngram: Maximum n-gram size to compute (LexSyn only)
            ignore_duplicates: Whether to ignore duplicate n-grams (LexSyn only)
            add_stanford_tags: Whether to include Stanford POS tags (LexSyn only)
            **kwargs: Additional arguments passed to the underlying analyzer
            
        Returns:
            pd.DataFrame: Concatenated results for all files
        """
        # If output directory is provided, ensure model-specific directory exists
        if output_directory:
            model_dir = os.path.join(output_directory, self.embedding_model)
            os.makedirs(model_dir, exist_ok=True)
        else:
            model_dir = None
        
        # Process files with the appropriate analyzer
        results = None
        
        if self.embedding_model == "bert":
            # BERT implementation
            results = self.analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=model_dir,  # Use model-specific directory
                file_pattern=file_pattern,
                lag=lag,
                **kwargs
            )
        elif self.embedding_model == "word2vec":
            # Word2Vec implementation
            results = self.analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=model_dir,  # Use model-specific directory
                file_pattern=file_pattern,
                lag=lag,
                high_sd_cutoff=high_sd_cutoff,
                low_n_cutoff=low_n_cutoff,
                save_vocab=save_vocab,
                **kwargs
            )
        elif self.embedding_model == "lexsyn":
            # LexSyn implementation
            results = self.analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=model_dir,  # Use model-specific directory
                file_pattern=file_pattern,
                lag=lag,
                max_ngram=max_ngram,
                ignore_duplicates=ignore_duplicates,
                add_stanford_tags=add_stanford_tags,
                **kwargs
            )
        
        # Optionally save as "real" results to distinguish from baseline
        if results is not None and not results.empty and model_dir:
            # Customize filename based on embedding model
            if self.embedding_model == "lexsyn":
                dup_str = "noDups" if ignore_duplicates else "withDups"
                stan_str = "withStan" if add_stanford_tags else "noStan"
                real_path = os.path.join(
                    model_dir,
                    f"real_alignment_lexsyn_ngram{max_ngram}_lag{lag}_{dup_str}_{stan_str}.csv"
                )
            elif self.embedding_model == "bert":
                real_path = os.path.join(
                    model_dir,
                    f"real_alignment_bert_lag{lag}.csv"
                )
            elif self.embedding_model == "word2vec":
                sd_str = f"sd{high_sd_cutoff}"
                n_str = f"n{low_n_cutoff}"
                real_path = os.path.join(
                    model_dir,
                    f"real_alignment_word2vec_lag{lag}_{sd_str}_{n_str}.csv"
                )
            
        return results
    
    def process_file(self, file_path, lag=1, high_sd_cutoff=3, low_n_cutoff=1, max_ngram=2,
                    ignore_duplicates=True, add_stanford_tags=False, **kwargs):
        """
        Process a single file to compute alignment metrics
        
        Args:
            file_path: Path to the file to process
            lag: Number of turns to lag when pairing utterances (default: 1)
            high_sd_cutoff: Standard deviation cutoff for high-frequency words (Word2Vec only)
            low_n_cutoff: Minimum frequency cutoff (Word2Vec only)
            max_ngram: Maximum n-gram size to compute (LexSyn only)
            ignore_duplicates: Whether to ignore duplicate n-grams (LexSyn only)
            add_stanford_tags: Whether to include Stanford POS tags (LexSyn only)
            **kwargs: Additional arguments passed to the underlying analyzer
            
        Returns:
            pd.DataFrame: Processed DataFrame with alignment metrics
        """
        if self.embedding_model == "bert":
            # BERT implementation
            return self.analyzer.process_file(file_path=file_path, lag=lag, **kwargs)
        elif self.embedding_model == "word2vec":
            # Word2Vec implementation
            return self.analyzer.process_file(
                file_path=file_path, 
                lag=lag, 
                high_sd_cutoff=high_sd_cutoff,
                low_n_cutoff=low_n_cutoff,
                **kwargs
            )
        elif self.embedding_model == "lexsyn":
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
                        all_surrogates=True, keep_original_turn_order=True, id_separator='\-',
                        condition_label='cond', dyad_label='dyad', lag=1, max_ngram=2,
                        high_sd_cutoff=3, low_n_cutoff=1, save_vocab=True,
                        ignore_duplicates=True, add_stanford_tags=False, **kwargs):
        """
        Generate surrogate conversation pairs and analyze their alignment as a baseline
        
        Args:
            input_files: Path to directory containing conversation files or list of file paths
            output_directory: Directory to save alignment results (default: "results")
            surrogate_directory: Directory to save surrogate files (optional)
            all_surrogates: Whether to generate all possible surrogate pairings (default: True)
            keep_original_turn_order: Whether to maintain original turn order (default: True)
            id_separator: Character separating dyad ID from condition ID (default: '\-')
            condition_label: String preceding condition ID in filenames (default: 'cond')
            dyad_label: String preceding dyad ID in filenames (default: 'dyad')
            lag: Number of turns to lag when analyzing alignment (default: 1)
            max_ngram: Maximum n-gram size for lexical/syntactic analysis (default: 2)
            high_sd_cutoff: Standard deviation cutoff for high-frequency words (Word2Vec only)
            low_n_cutoff: Minimum frequency cutoff (Word2Vec only)
            save_vocab: Whether to save vocabulary lists (Word2Vec only)
            ignore_duplicates: Whether to ignore duplicate n-grams (LexSyn only)
            add_stanford_tags: Whether to include Stanford POS tags (default: False)
            **kwargs: Additional arguments for alignment analysis
            
        Returns:
            pd.DataFrame: Baseline alignment results for surrogate pairs
        """
        # Create a SurrogateAlignment instance with same embedding model
        surrogate_aligner = SurrogateAlignment(embedding_model=self.embedding_model)
        
        # Pass through all parameters to the surrogate analyzer
        return surrogate_aligner.analyze_baseline(
            input_files=input_files,
            output_directory=output_directory,
            surrogate_directory=surrogate_directory,
            all_surrogates=all_surrogates,
            keep_original_turn_order=keep_original_turn_order,
            id_separator=id_separator,
            condition_label=condition_label,
            dyad_label=dyad_label,
            lag=lag,
            max_ngram=max_ngram,
            high_sd_cutoff=high_sd_cutoff,
            low_n_cutoff=low_n_cutoff,
            save_vocab=save_vocab,
            ignore_duplicates=ignore_duplicates,
            add_stanford_tags=add_stanford_tags,
            **kwargs
        )