import os
from pathlib import Path
import json
import warnings

CONFIG_DIR = Path.home() / ".config" / "my_package"
CONFIG_FILE = CONFIG_DIR / "config.json"

def get_huggingface_token(token=None):
    """
    Get Hugging Face token with the following priority:
    1. Explicitly passed token
    2. Environment variable HUGGINGFACE_TOKEN
    3. Config file at ~/.config/my_package/config.json
    4. Fall back to None with a warning
    """
    # 1. Use provided token if available
    if token:
        return token
    
    # 2. Check environment variable
    env_token = os.environ.get("HUGGINGFACE_TOKEN")
    if env_token:
        return env_token
    
    # 3. Check config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                if "huggingface_token" in config:
                    return config["huggingface_token"]
        except (json.JSONDecodeError, OSError):
            pass  # Handle silently and continue to fallback
    
    # 4. No token found
    warnings.warn(
        "No Hugging Face token found. Some functionality will be limited. "
        "Set the HUGGINGFACE_TOKEN environment variable or create a config file."
    )
    return None

def save_token_to_config(token):
    """Save token to config file for future use"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    
    config["huggingface_token"] = token
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    
    return True