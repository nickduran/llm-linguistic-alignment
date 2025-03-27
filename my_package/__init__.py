# my_package/__init__.py
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from .bert_model import BertWrapper
from .fasttext_model import FastTextWrapper
from .alignment_bert import SemanticAlignmentAnalyzer
from .alignment_fasttext import SemanticAlignmentFastText
from .alignment_lexsyn import LexicalSyntacticAlignment
from .surrogates import SurrogateGenerator, SurrogateAlignment
from .alignment import SemanticAlignment

__all__ = ['BertWrapper', 'FastTextWrapper', 'SemanticAlignmentAnalyzer', 
           'SemanticAlignmentFastText', 'LexicalSyntacticAlignment', 'SemanticAlignment',
           'SurrogateGenerator', 'SurrogateAlignment']

# from .prepare_transcripts import *
# from .calculate_alignment import *
# from . import datasets
# __version__ = "0.1.1"