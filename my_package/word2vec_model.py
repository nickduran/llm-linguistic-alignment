# my_package/word2vec_model.py
import os
import numpy as np
import gensim
import gensim.downloader as api
import pickle

class Word2VecWrapper:
    def __init__(self, model_name="word2vec-google-news-300", cache_dir=None):
        """
        Initialize Word2Vec model with caching
        
        Args:
            model_name: Name of the Word2Vec model to use
            cache_dir: Directory to cache models (optional)
        """
        self.model_name = model_name
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "my_package", "models")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Configure Gensim to use the cache directory
        api.BASE_DIR = cache_dir
        
        # Initialize embedding cache
        self.embedding_cache_path = os.path.join(self.cache_dir, f"{self.model_name}_embedding_cache.pkl")
        self.embedding_cache = self._load_embedding_cache()
        
        # Load the model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load Word2Vec model, checking global namespace first
        
        Returns:
            gensim.models.KeyedVectors: Loaded model
        """
        # Check if model is already loaded in global namespace
        global_vars = globals()
        model_var_name = f"{self.model_name.replace('-', '_')}_model"
        
        if model_var_name in global_vars and global_vars[model_var_name] is not None:
            print(f"Using {self.model_name} model from global namespace")
            return global_vars[model_var_name]
        
        # Try to load from gensim
        try:
            print(f"Loading model: {self.model_name}")
            model = api.load(self.model_name)
            print(f"Model loaded: {self.model_name}")
            
            # Store in global namespace for future use
            global_vars[model_var_name] = model
            
            return model
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
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
                print(f"Error loading embedding cache: {e}")
        
        return {}
    
    def save_embedding_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            print(f"Error saving embedding cache: {e}")
    
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