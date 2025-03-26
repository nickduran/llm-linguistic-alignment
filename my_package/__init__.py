import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from .bert_model import BertWrapper
from .word2vec_model import Word2VecWrapper
from .alignment_bert import SemanticAlignmentAnalyzer
from .alignment_w2v import SemanticAlignmentW2V
from .alignment_lexsyn import LexicalSyntacticAlignment
from .alignment import SemanticAlignment

__all__ = ['BertWrapper', 'Word2VecWrapper', 'SemanticAlignmentAnalyzer', 
           'SemanticAlignmentW2V', 'LexicalSyntacticAlignment', 'SemanticAlignment']

# from .prepare_transcripts import *
# from .calculate_alignment import *
# from . import datasets
# __version__ = "0.1.1"