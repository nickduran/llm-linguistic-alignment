# my_package/__init__.py
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from .bert_model import BertWrapper
from .fasttext_model import Word2VecWrapper
from .alignment_bert import SemanticAlignmentAnalyzer
from .alignment_w2v import SemanticAlignmentW2V
from .alignment_lexsyn import LexicalSyntacticAlignment
from .surrogates import SurrogateGenerator, SurrogateAlignment
from .alignment import SemanticAlignment

__all__ = ['BertWrapper', 'Word2VecWrapper', 'SemanticAlignmentAnalyzer', 
           'SemanticAlignmentW2V', 'LexicalSyntacticAlignment', 'SemanticAlignment',
           'SurrogateGenerator', 'SurrogateAlignment']

# from .prepare_transcripts import *
# from .calculate_alignment import *
# from . import datasets
# __version__ = "0.1.1"