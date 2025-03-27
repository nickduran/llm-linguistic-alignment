# my_package/fasttext_model.py
import os
import numpy as np
import gensim
import gensim.downloader as api
import pickle
import warnings
import urllib.request
import zipfile

class FastTextWrapper:
    def __init__(self, model_name="fasttext-wiki-news-300", cache_dir=None):
        """
        Initialize FastText model with caching
        
        Args:
            model_name: Name of the FastText model to use
            cache_dir: Directory to cache models (optional)
        """
        self.model_name = model_name
        
        # Set up cache directory
        if cache_dir is None:
            # Default location
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "align", "models")
        
        # Create the cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Set Gensim's download directory
        api.BASE_DIR = self.cache_dir
        
        # Initialize embedding cache
        self.embedding_cache_path = os.path.join(self.cache_dir, f"{self.model_name}_embedding_cache.pkl")
        self.embedding_cache = self._load_embedding_cache()
        
        # Load the model
        self.model = self._load_model()
        
        # Print cache information
        print(f"Using model cache directory: {self.cache_dir}")
        print(f"Using embedding cache: {self.embedding_cache_path}")
    
    def _load_model(self):
        """
        Load FastText model from Gensim with simplified download handling
        
        Returns:
            gensim.models.KeyedVectors: Loaded model
        """
        try:
            print(f"Loading model: {self.model_name}")
            print(f"Using model cache directory: {self.cache_dir}")
            
            # Define the exact path where we want to store the model
            model_file_path = os.path.join(self.cache_dir, f"{self.model_name}.kv")
            model_gz_path = os.path.join(self.cache_dir, f"{self.model_name}.gz")
            
            # Check if we already have the model in KeyedVectors format
            if os.path.exists(model_file_path):
                print(f"Loading cached model from: {model_file_path}")
                return gensim.models.KeyedVectors.load(model_file_path)
                
            # Check if we have the raw .gz file
            elif os.path.exists(model_gz_path):
                print(f"Loading model from: {model_gz_path}")
                model = gensim.models.KeyedVectors.load_word2vec_format(model_gz_path, binary=True)
                # Save in our preferred format for future use
                print(f"Saving model to cache: {model_file_path}")
                model.save(model_file_path)
                return model
            
            # If model doesn't exist locally, download it directly
            else:
                print("Model not found in cache. Downloading directly...")
                
                # Create directory if it doesn't exist
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # For fasttext-wiki-news-300, we know the direct URL
                if self.model_name == "fasttext-wiki-news-300":
                    
                    # Define the URL (this is a widely available mirror)
                    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
                    
                    # Simplified download code
                    print(f"Downloading from {url}...")
                    temp_zip_path = os.path.join(self.cache_dir, "temp_model.zip")
                    urllib.request.urlretrieve(url, temp_zip_path)
                    print("Download complete.")
                    
                    # Extract and load
                    print("Extracting model file...")
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        # Extract to our cache directory
                        zip_ref.extractall(self.cache_dir)
                    
                    # Remove the zip file
                    os.remove(temp_zip_path)
                    
                    # Find the extracted file (should be a .vec file)
                    vec_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.vec')]
                    if vec_files:
                        vec_path = os.path.join(self.cache_dir, vec_files[0])
                        print(f"Loading model from: {vec_path}")
                        # Load the model from the text format
                        model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
                        # Save in our preferred format
                        print(f"Saving model to cache: {model_file_path}")
                        model.save(model_file_path)
                        return model
                    else:
                        raise FileNotFoundError("Could not find extracted model file.")
                else:
                    # For other models, we can use Gensim's API
                    try:
                        # Try using Gensim's downloader API as fallback
                        print(f"Direct download not implemented for {self.model_name}, trying Gensim API")
                        model = api.load(self.model_name)
                        print(f"Model loaded successfully: {self.model_name}")
                        
                        # Save to our cache location for future use
                        print(f"Saving model to cache: {model_file_path}")
                        model.save(model_file_path)
                        return model
                    except Exception as e:
                        print(f"Gensim API failed: {e}")
                        raise ValueError(f"Could not download model: {self.model_name}")
                    
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Detailed error information:")
            import traceback
            traceback.print_exc()
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
        if self.model is None:
            return None
            
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
        
        if tokens is None or not tokens or self.model is None:
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
        
        return result