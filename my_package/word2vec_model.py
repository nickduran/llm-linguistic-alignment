# my_package/word2vec_model.py
import os
import numpy as np
import gensim
import gensim.downloader as api
from .config import get_huggingface_token

class Word2VecWrapper:
    def __init__(self, model_name="word2vec-google-news-300", cache_dir=None, token=None):
        """
        Initialize Word2Vec model with caching
        
        Args:
            model_name: Name of the Word2Vec model to use
            cache_dir: Directory to cache models (optional)
            token: Hugging Face token (optional)
        """
        self.model_name = model_name
        self.token = get_huggingface_token(token)
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "my_package", "models")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Configure Gensim to use the cache directory
        api.BASE_DIR = cache_dir
        
        # Load the model
        self.model = self._load_model()
        
        # Initialize embedding cache
        self.embedding_cache = {}
    
    def _load_model(self):
        """Load Word2Vec model from Gensim or local cache"""
        try:
            print(f"Loading model: {self.model_name}")
            model = api.load(self.model_name)
            print(f"Model loaded: {self.model_name}")
            return model
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            return None
    
    def get_word_embedding(self, word):
        """Get embedding for a single word"""
        if word in self.model.key_to_index:
            return self.model[word]
        return None
    
    def get_text_embedding(self, tokens):
        """
        Get embedding for a list of tokens by averaging word vectors
        
        Args:
            tokens: List of tokens to encode
            
        Returns:
            numpy.ndarray: Embedding vector or None if no tokens can be encoded
        """
        if tokens is None or not tokens:
            return None
            
        # Get embeddings for words that are in the vocabulary
        embeddings = []
        for word in tokens:
            if word in self.model.key_to_index:
                embeddings.append(self.model[word])
        
        if not embeddings:
            return None
            
        # Average the embeddings
        return np.mean(embeddings, axis=0)
    
    def cache_embedding(self, text, embedding):
        """Add an embedding to the cache"""
        self.embedding_cache[text] = embedding
        
    def get_cached_embedding(self, text):
        """Get an embedding from the cache if available"""
        return self.embedding_cache.get(text)
    
    def save_cache(self, cache_path):
        """Save the embedding cache to a file"""
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        
    def load_cache(self, cache_path):
        """Load the embedding cache from a file"""
        import pickle
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)