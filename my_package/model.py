from transformers import AutoTokenizer, AutoModel
from .config import get_huggingface_token

class BertWrapper:
    def __init__(self, model_name="bert-base-uncased", token=None):
        """
        Initialize BERT model with token handling
        
        Args:
            model_name: Name of the BERT model to use
            token: Hugging Face token (optional)
        """
        self.token = get_huggingface_token(token)
        
        # Update tokenizer initialization - remove the problematic parameter
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=self.token
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            token=self.token
        )
    
    def encode(self, text):
        """Encode text using BERT"""
        tokens = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**tokens)
        return outputs.last_hidden_state