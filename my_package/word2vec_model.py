# my_package/word2vec_model.py
import os
import numpy as np
import gensim
import gensim.downloader as api
import pickle
import warnings

class Word2VecWrapper:
    def __init__(self, model_name="word2vec-google-news-300", cache_dir=None):
        """
        Initialize Word2Vec model with caching
        
        Args:
            model_name: Name of the Word2Vec model to use
            cache_dir: Directory to cache models (optional)
        """
        self.model_name = model_name
        
        # Set default cache directory if none provided
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "align", "models")
        
        # Create directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Tell Gensim to use this directory for downloading models
        api.BASE_DIR = self.cache_dir
        
        # Load the model (calling another method)
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load Word2Vec model from Gensim or local cache
        
        Returns:
            gensim.models.KeyedVectors: Loaded model
        """
        try:
            print(f"Loading model: {self.model_name}")
            # Check if model directory exists
            model_path = os.path.join(self.gensim_data_dir, self.model_name)
            
            if os.path.exists(model_path):
                print(f"Model directory exists: {model_path}")
                
                # For common models, check specific file paths
                if self.model_name == "word2vec-google-news-300":
                    bin_path = os.path.join(model_path, f"{self.model_name}.gz")
                    if os.path.exists(bin_path):
                        print(f"Loading model from: {bin_path}")
                        return gensim.models.KeyedVectors.load_word2vec_format(bin_path, binary=True)
            
            # If not found in cache or error in loading, download via API
            print(f"Downloading model via gensim API to: {self.gensim_data_dir}")
            model = api.load(self.model_name)
            print(f"Model loaded successfully: {self.model_name}")
            return model
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            warnings.warn(f"Failed to load model {self.model_name}. Analysis will proceed but may not work correctly.")
            return None
    
    def _load_embedding_cache(self):
        """
        Load embedding cache from disk if it exists
        
        Returns:
            dict: Embedding cache
        """
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded embedding cache with {len(cache)} entries")
                return cache
            except Exception as e:
                print(f"Error loading embedding cache: {str(e)}")
        
        return {}
    
    def save_embedding_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
    
    def get_word_embedding(self, word):
        """Get embedding for a single word"""
        if word in self.model.key_to_index:
            return self.model[word]
        return None
    
    def get_text_embedding(self, tokens, cache_key=None):
        """
        Get embedding for a list of tokens by averaging word vectors
        
        Args:
            tokens: List of tokens to encode
            cache_key: Optional key for caching the result
            
        Returns:
            numpy.ndarray: Embedding vector or None if no tokens can be encoded
        """
        # Check cache if cache_key is provided
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if tokens is None or not tokens:
            return None
            
        # Get embeddings for words that are in the vocabulary
        embeddings = []
        for word in tokens:
            # Skip non-string tokens
            if not isinstance(word, str):
                continue
                
            # Try to get embedding from model
            if word in self.model.key_to_index:
                embeddings.append(self.model[word])
        
        if not embeddings:
            return None
            
        # Average the embeddings
        result = np.mean(embeddings, axis=0)
        
        # Cache the result if a cache_key was provided
        if cache_key:
            self.embedding_cache[cache_key] = result
            
            # Periodically save the cache to disk (e.g., every 100 new entries)
            if len(self.embedding_cache) % 100 == 0:
                self.save_embedding_cache()
        
        return result