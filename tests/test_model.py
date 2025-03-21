# tests/test_model.py
import os
import pytest
from my_package.bert_model import BertWrapper

def test_bert_with_env_token():
    """Test that BERT works with environment token"""
    # Skip if no token available in environment
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        pytest.skip("No HF token in environment")
    
    model = BertWrapper()
    output = model.encode("Hello world")
    assert output is not None
    # assert output.shape[1] == 768  # BERT base hidden size

    # Print the shape to debug
    print(f"Full output shape: {output.shape}")
    
    # The correct assertion depends on what you want to check:
    # output.shape should be [1, sequence_length, 768]
    # So for BERT base, output.shape[2] should be 768 (the hidden dimension)
    assert output.shape[2] == 768  # Check hidden dimension size
    
def test_bert_with_explicit_token():
    """Test that BERT works with explicitly provided token"""
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        pytest.skip("No HF token in environment")
    
    model = BertWrapper(token=token)
    output = model.encode("Hello world")
    assert output is not None