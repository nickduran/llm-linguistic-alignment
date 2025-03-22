# my_package/alignment.py
from .alignment_bert import SemanticAlignmentAnalyzer
from .alignment_w2v import SemanticAlignmentW2V

class SemanticAlignment:
    def __init__(self, embedding_model="bert", model_name=None, token=None, cache_dir=None):
        """
        Initialize a semantic alignment analyzer with a specified embedding model
        
        Args:
            embedding_model: Type of embedding model to use ('bert' or 'word2vec')
            model_name: Name of the specific model to use (optional)
            token: API token for model access (optional)
            cache_dir: Directory to cache models and embeddings (optional)
        """
        self.embedding_model = embedding_model.lower()
        
        if self.embedding_model == "bert":
            model_name = model_name or "bert-base-uncased"
            self.analyzer = SemanticAlignmentAnalyzer(model_name=model_name, token=token)
        elif self.embedding_model == "word2vec":
            model_name = model_name or "word2vec-google-news-300"
            self.analyzer = SemanticAlignmentW2V(model_name=model_name, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}. Use 'bert' or 'word2vec'.")
    
    def analyze_folder(self, folder_path, output_directory=None, file_pattern="*.txt", lag=1, 
                       high_sd_cutoff=3, low_n_cutoff=1, save_vocab=True, **kwargs):
        """
        Analyze semantic alignment for all text files in a folder
        
        Args:
            folder_path: Path to folder containing text files
            output_directory: Directory to save results (optional)
            file_pattern: Pattern to match text files (default: "*.txt")
            lag: Number of turns to lag when pairing utterances (default: 1)
            high_sd_cutoff: Standard deviation cutoff for high-frequency words (Word2Vec only)
            low_n_cutoff: Minimum frequency cutoff (Word2Vec only)
            save_vocab: Whether to save vocabulary lists (Word2Vec only)
            **kwargs: Additional arguments passed to the underlying analyzer
            
        Returns:
            pd.DataFrame: Concatenated results for all files
        """
        if self.embedding_model == "bert":
            # BERT implementation - ignore Word2Vec-specific parameters
            return self.analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=output_directory,
                file_pattern=file_pattern,
                lag=lag,
                **kwargs
            )
        elif self.embedding_model == "word2vec":
            # Word2Vec implementation - pass all parameters
            return self.analyzer.analyze_folder(
                folder_path=folder_path,
                output_directory=output_directory,
                file_pattern=file_pattern,
                lag=lag,
                high_sd_cutoff=high_sd_cutoff,
                low_n_cutoff=low_n_cutoff,
                save_vocab=save_vocab,
                **kwargs
            )
    
    def process_file(self, file_path, lag=1, high_sd_cutoff=3, low_n_cutoff=1, **kwargs):
        """
        Process a single file to compute embeddings and similarities
        
        Args:
            file_path: Path to the file to process
            lag: Number of turns to lag when pairing utterances (default: 1)
            high_sd_cutoff: Standard deviation cutoff for high-frequency words (Word2Vec only)
            low_n_cutoff: Minimum frequency cutoff (Word2Vec only)
            **kwargs: Additional arguments passed to the underlying analyzer
            
        Returns:
            pd.DataFrame: Processed DataFrame with embeddings and similarities
        """
        if self.embedding_model == "bert":
            # BERT implementation - ignore Word2Vec-specific parameters
            return self.analyzer.process_file(file_path=file_path, lag=lag, **kwargs)
        elif self.embedding_model == "word2vec":
            # Word2Vec implementation - pass all parameters
            return self.analyzer.process_file(
                file_path=file_path, 
                lag=lag, 
                high_sd_cutoff=high_sd_cutoff,
                low_n_cutoff=low_n_cutoff,
                **kwargs
            )