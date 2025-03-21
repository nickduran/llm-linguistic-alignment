import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from .model import BertWrapper
from .alignment_cache import SemanticAlignmentAnalyzer

__all__ = ['BertWrapper', 'SemanticAlignmentAnalyzer']

# from .prepare_transcripts import *
# from .calculate_alignment import *
# from . import datasets
# __version__ = "0.1.1"