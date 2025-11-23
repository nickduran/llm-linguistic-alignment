"""
prepare_transcripts.py - Transcript Preprocessing Module for ALIGN Package

This module prepares raw conversational transcript files for linguistic alignment analysis.
It performs text cleaning, tokenization, lemmatization, and part-of-speech tagging.

Key Features:
- Text cleaning with filler removal
- Adjacent turn merging
- Optional spell-checking with Bayesian algorithm
- Tokenization and lemmatization
- Multiple POS tagging options (NLTK, spaCy, Stanford)
- Batch processing of multiple files
- Output format compatible with alignment analysis

Author: Nicholas Duran
Modified: 2025 (refactored for improved compatibility and performance)
"""

import os
import re
import string
import glob
from collections import Counter, defaultdict, OrderedDict

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from nltk.util import ngrams

from tqdm import tqdm

# Optional: spaCy for faster POS tagging
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

def ensure_nltk_resources():
    """
    Ensure required NLTK resources are downloaded.
    
    Downloads resources automatically if they're not found, so users don't
    need to manually run nltk.download().
    """
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')  # Required for WordNet lemmatizer
    ]
    
    missing_resources = []
    
    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing_resources.append(resource_name)
    
    if missing_resources:
        print("Downloading required NLTK resources...")
        for resource in missing_resources:
            print(f"  - Downloading {resource}...")
            try:
                nltk.download(resource, quiet=True)
                print(f"    ✓ {resource} downloaded successfully")
            except Exception as e:
                print(f"    ✗ Failed to download {resource}: {e}")
        print("NLTK resources ready!\n")
    else:
        print("✓ All required NLTK resources are available\n")


def InitialCleanup(dataframe,
                   minwords=2,
                   use_filler_list=None,
                   filler_regex_and_list=False):
    """
    Perform basic text cleaning to prepare dataframe for analysis.
    
    Removes non-letter/-space characters, empty turns, turns below a minimum 
    length, and fillers.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe with 'participant' and 'content' columns
    minwords : int, optional
        Minimum number of words required per turn (default: 2)
    use_filler_list : list of str, optional
        List of filler words to remove. If None, uses regex to remove 
        common fillers (default: None)
    filler_regex_and_list : bool, optional
        If True, use both regex and custom filler list (default: False)

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 'content' column processed
    """

    # Only allow strings, spaces, apostrophes, and newlines to pass
    WHITELIST = string.ascii_letters + '\'' + ' '

    # Remove inadvertent empty turns
    dataframe = dataframe[pd.notnull(dataframe['content'])]
    dataframe = dataframe.dropna(subset=['content'])
    
    # Internal function: remove fillers via regular expressions
    def applyRegExpression(textFiller):
        # Remove common speech fillers while preserving words like "mom", "am", "ham"
        textClean = re.sub(r'^(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]+\s', ' ', textFiller)
        textClean = re.sub(r'\s(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]+\s', ' ', textClean)
        textClean = re.sub(r'\s(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]$', ' ', textClean)
        textClean = re.sub(r'^(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]$', ' ', textClean)
        return textClean
    
    # Create a new column with only approved text
    dataframe['clean_content'] = dataframe['content'].apply(
        lambda utterance: ''.join([char for char in utterance if char in WHITELIST]).lower()
    )

    # Apply filler removal based on user preferences
    if use_filler_list is None and not filler_regex_and_list:
        # DEFAULT: remove typical speech fillers via regex
        dataframe['clean_content'] = dataframe['clean_content'].apply(applyRegExpression)
    elif use_filler_list is not None and not filler_regex_and_list:
        # OPTION 1: remove only user-specified fillers
        dataframe['clean_content'] = dataframe['clean_content'].apply(
            lambda utterance: ' '.join([word for word in utterance.split(" ") if word not in use_filler_list])
        )
    elif use_filler_list is not None and filler_regex_and_list:
        # OPTION 2: remove both regex fillers and user-specified fillers
        dataframe['clean_content'] = dataframe['clean_content'].apply(applyRegExpression)
        dataframe['clean_content'] = dataframe['clean_content'].apply(
            lambda utterance: ' '.join([word for word in utterance.split(" ") if word not in use_filler_list])
        )

    # Drop the old "content" column and rename the clean version
    dataframe = dataframe.drop(['content'], axis=1)
    dataframe = dataframe.rename(columns={'clean_content': 'content'})

    # Remove rows that don't meet minwords requirement
    dataframe['utteranceLen'] = dataframe['content'].apply(lambda x: word_tokenize(x)).str.len()
    dataframe = dataframe.drop(dataframe[dataframe.utteranceLen < int(minwords)].index)
    dataframe = dataframe.drop(['utteranceLen'], axis=1)
    dataframe = dataframe.reset_index(drop=True)

    return dataframe


def AdjacentMerge(dataframe):
    """
    Merge adjacent turns by the same speaker.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with 'participant' and 'content' columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with adjacent turns by same speaker merged
    """

    repeat = 1
    while repeat == 1:
        l1 = len(dataframe)
        DfMerge = []
        k = 0
        if len(dataframe) > 0:
            while k < len(dataframe) - 1:
                if dataframe['participant'].iloc[k] != dataframe['participant'].iloc[k+1]:
                    DfMerge.append([dataframe['participant'].iloc[k], dataframe['content'].iloc[k]])
                    k = k + 1
                elif dataframe['participant'].iloc[k] == dataframe['participant'].iloc[k+1]:
                    DfMerge.append([
                        dataframe['participant'].iloc[k], 
                        dataframe['content'].iloc[k] + " " + dataframe['content'].iloc[k+1]
                    ])
                    k = k + 2
            if k == len(dataframe) - 1:
                DfMerge.append([dataframe['participant'].iloc[k], dataframe['content'].iloc[k]])

        dataframe = pd.DataFrame(DfMerge, columns=('participant', 'content'))
        if l1 == len(dataframe):
            repeat = 0

    return dataframe


def Tokenize(text):
    """
    Tokenize text and expand common contractions.
    
    Parameters
    ----------
    text : str
        Text to tokenize
        
    Returns
    -------
    list of str
        Tokenized text with contractions expanded
    """

    # Dictionary of common contractions
    contract_dict = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot",
        "can't've": "cannot have", "'cause": "because", "could've": "could have",
        "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
        "he'd": "he had", "he'd've": "he would have", "he'll": "he will",
        "he'll've": "he will have", "he's": "he is", "how'd": "how did",
        "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
        "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
        "i'll've": "i will have", "i'm": "i am", "i've": "i have",
        "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is",
        "let's": "let us", "ma'am": "madam", "mayn't": "may not",
        "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
        "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
        "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
        "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
        "she's": "she is", "should've": "should have", "shouldn't": "should not",
        "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
        "that'd": "that had", "that'd've": "that would have", "that's": "that is",
        "there'd": "there would", "there'd've": "there would have", "there's": "there is",
        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
        "they'll've": "they will have", "they're": "they are", "they've": "they have",
        "to've": "to have", "wasn't": "was not", "we'd": "we would",
        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
        "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what'll've": "what will have", "what're": "what are",
        "what's": "what is", "what've": "what have", "when's": "when is",
        "when've": "when have", "where'd": "where did", "where's": "where is",
        "where've": "where have", "who'll": "who will", "who'll've": "who will have",
        "who's": "who is", "who've": "who have", "why's": "why is",
        "why've": "why have", "will've": "will have", "won't": "will not",
        "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
        "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
        "y'all'd've": "you all would have", "y'all're": "you all are",
        "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
        "you'll": "you will", "you'll've": "you will have", "you're": "you are",
        "you've": "you have"
    }
    
    contractions_re = re.compile('(%s)' % '|'.join(list(contract_dict.keys())))

    # Internal function to expand contractions
    def expand_contractions(text, contractions_re=contractions_re):
        def replace(match):
            return contract_dict[match.group(0)]
        return contractions_re.sub(replace, text.lower())

    # Process text
    text = expand_contractions(text)
    cleantoken = word_tokenize(text)
            
    return cleantoken


def TokenizeSpell(text, nwords):
    """
    Tokenize text with spell-checking.
    
    Uses a Bayesian spell-checking algorithm to correct spelling errors.
    Based on: http://norvig.com/spell-correct.html

    Parameters
    ----------
    text : str
        Text to tokenize and spell-check
    nwords : dict
        Dictionary of known words with frequencies (from training corpus)
        
    Returns
    -------
    list of str
        Tokenized and spell-checked text
    """
    
    # Internal function: identify possible spelling errors
    def edits1(word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in string.ascii_lowercase if b]
        inserts = [a + c + b for a, b in splits for c in string.ascii_lowercase]
        return set(deletes + transposes + replaces + inserts)

    # Internal function: identify known edits at edit distance 2
    def known_edits2(word, nwords):
        return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in nwords)

    # Internal function: identify known words
    def known(words, nwords):
        return set(w for w in words if w in nwords)

    # Internal function: correct spelling
    def correct(word, nwords):
        candidates = known([word], nwords) or known(edits1(word), nwords) or known_edits2(word, nwords) or [word]
        return max(candidates, key=nwords.get)
    
    cleantoken = []
    token = Tokenize(text)
        
    for word in token:
        # Don't spell-check words with apostrophes (contractions, possessives)
        if "'" not in word:
            cleantoken.append(correct(word, nwords))
        else:
            cleantoken.append(word)
    
    return cleantoken


def pos_to_wn(tag):
    """
    Convert Penn Treebank POS tag to WordNet format for lemmatization.
    
    Parameters
    ----------
    tag : str
        Penn Treebank POS tag
        
    Returns
    -------
    str
        WordNet POS tag (NOUN, VERB, ADV, or ADJ)
    """

    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']
    
    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    
    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']
    
    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']

    if is_noun(tag):
        return wn.NOUN
    elif is_verb(tag):
        return wn.VERB
    elif is_adverb(tag):
        return wn.ADV
    elif is_adjective(tag):
        return wn.ADJ
    else:
        return wn.NOUN


def Lemmatize(tokenlist):
    """
    Lemmatize a list of tokens using WordNet.
    
    Parameters
    ----------
    tokenlist : list of str
        Tokens to lemmatize
        
    Returns
    -------
    list of str
        Lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    defaultPos = nltk.pos_tag(tokenlist)
    words_lemma = []
    for item in defaultPos:
        words_lemma.append(lemmatizer.lemmatize(item[0], pos_to_wn(item[1])))
    return words_lemma


def batch_stanford_tag(token_lists, stanford_tagger, batch_size=50):
    """
    Tag multiple utterances with Stanford tagger in batches for better performance.
    
    Processing utterances in batches reduces Java startup overhead significantly.
    
    Parameters
    ----------
    token_lists : list of lists
        List of tokenized utterances to tag
    stanford_tagger : StanfordPOSTagger
        Initialized Stanford POS tagger
    batch_size : int, optional
        Number of utterances to process in each batch (default: 50)
        
    Returns
    -------
    list of str
        String representations of tagged utterances, compatible with ast.literal_eval()
    """
    results = []
    
    print(f"Processing {len(token_lists)} utterances in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(token_lists), batch_size), desc="Stanford tagging"):
        batch = token_lists[i:i+batch_size]
        
        # Flatten batch into single list with boundary markers
        flattened = []
        boundaries = [0]  # Start positions of each utterance
        
        for tokens in batch:
            if tokens:  # Only process non-empty token lists
                flattened.extend(tokens)
                boundaries.append(len(flattened))
        
        # Tag the entire batch at once
        if flattened:
            try:
                tagged_flat = stanford_tagger.tag(flattened)
                
                # Split back into individual utterances using boundaries
                for j in range(len(boundaries) - 1):
                    start_idx = boundaries[j]
                    end_idx = boundaries[j + 1]
                    utterance_tags = tagged_flat[start_idx:end_idx]
                    results.append(str(utterance_tags))
            except Exception as e:
                print(f"\nWarning: Stanford tagging failed for batch {i//batch_size + 1}: {e}")
                # Add empty results for this batch
                for _ in batch:
                    results.append(str([]))
        else:
            # Add empty results for empty batches
            for _ in batch:
                results.append(str([]))
    
    return results


def initialize_spacy_tagger(model_name='en_core_web_sm'):
    """
    Initialize spaCy tagger with automatic model download if needed.
    
    Parameters
    ----------
    model_name : str, optional
        spaCy model to use (default: 'en_core_web_sm')
        
    Returns
    -------
    spacy.Language
        Loaded spaCy model with only tagger pipeline
    """
    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is not installed. Install it with:\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        )
    
    try:
        nlp = spacy.load(model_name)
        print(f"Loaded spaCy model: {model_name}")
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading...")
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', model_name], check=True)
        nlp = spacy.load(model_name)
        print(f"Successfully downloaded and loaded: {model_name}")
    
    # Disable unnecessary pipeline components for speed
    pipes_to_disable = [pipe for pipe in nlp.pipe_names if pipe not in ['tagger', 'tok2vec']]
    nlp.disable_pipes(pipes_to_disable)
    print(f"spaCy tagger initialized (disabled: {pipes_to_disable})")
    
    return nlp


def spacy_tag_tokens(tokens, nlp):
    """
    Tag a list of tokens using spaCy and return Penn Treebank tags.
    
    Parameters
    ----------
    tokens : list of str
        Tokens to tag
    nlp : spacy.Language
        Loaded spaCy model
        
    Returns
    -------
    list of tuple
        List of (token, tag) tuples using Penn Treebank tagset
    """
    if not tokens:
        return []
    
    # Create a Doc from pre-tokenized text
    from spacy.tokens import Doc
    doc = Doc(nlp.vocab, words=tokens)
    
    # Process through the pipeline to get POS tags
    # This runs the enabled pipeline components (tagger, etc.)
    for name, pipe in nlp.pipeline:
        doc = pipe(doc)
    
    # Return tags in same format as NLTK: [('word', 'TAG'), ...]
    # Use .tag_ for Penn Treebank tags (compatible with NLTK/Stanford)
    return [(token.text, token.tag_) for token in doc]


def ApplyPOSTagging(df,
                    filename,
                    add_additional_tags=False,
                    tagger_type='stanford',
                    stanford_pos_path=None,
                    stanford_language_path=None,
                    stanford_batch_size=50,
                    spacy_model='en_core_web_sm'):
    """
    Apply part-of-speech tagging to dataframe.
    
    NLTK Penn Treebank tagger is always applied. Optionally, add a second tagger
    (Stanford or spaCy) for additional analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'token' and 'lemma' columns
    filename : str
        Name of source file (added to output)
    add_additional_tags : bool, optional
        Whether to add a second set of POS tags using an alternative tagger
        (default: False)
    tagger_type : str, optional
        Which tagger to use for additional tags:
        - 'stanford': Stanford CoreNLP tagger → creates tagged_stan_token, tagged_stan_lemma
        - 'spacy': spaCy tagger → creates tagged_spacy_token, tagged_spacy_lemma
        (default: 'stanford' for backward compatibility)
    stanford_pos_path : str, optional
        Path to Stanford POS tagger directory (required if stanford_tagger_type='stanford')
    stanford_language_path : str, optional
        Relative path to language model within Stanford directory
        (e.g., 'models/english-left3words-distsim.tagger')
    stanford_batch_size : int, optional
        Batch size for Stanford tagging (default: 50)
    spacy_model : str, optional
        spaCy model to use (default: 'en_core_web_sm')
        
    Returns
    -------
    pd.DataFrame
        Dataframe with POS tagging columns added, all stored as string 
        representations
        
    Examples
    --------
    # NLTK only (fastest)
    >>> df = ApplyPOSTagging(df, 'file.txt', add_additional_tags=False)
    
    # NLTK + spaCy (recommended)
    >>> df = ApplyPOSTagging(df, 'file.txt', add_additional_tags=True, 
    ...                      tagger_type='spacy')
    
    # NLTK + Stanford (slowest, highest accuracy)
    >>> df = ApplyPOSTagging(df, 'file.txt', add_additional_tags=True,
    ...                      tagger_type='stanford',
    ...                      stanford_pos_path='/path/to/stanford/',
    ...                      stanford_language_path='models/english-left3words-distsim.tagger')
    """
    
    # Initialize taggers if needed
    stanford_tagger = None
    spacy_nlp = None
    
    if add_additional_tags:
        if tagger_type == 'stanford':
            # Validate Stanford parameters
            if stanford_pos_path is None or stanford_language_path is None:
                raise ValueError(
                    'To use Stanford POS tagger, you must specify both:\n'
                    '  - stanford_pos_path: path to Stanford POS tagger directory\n'
                    '  - stanford_language_path: path to language model (relative to stanford_pos_path)\n'
                    'Example:\n'
                    '  stanford_pos_path="/path/to/stanford-postagger-full-2020-11-17/"\n'
                    '  stanford_language_path="models/english-left3words-distsim.tagger"'
                )
            
            # Normalize paths for cross-platform compatibility
            stanford_pos_path = os.path.normpath(stanford_pos_path)
            
            # Construct full paths
            model_path = os.path.join(stanford_pos_path, stanford_language_path)
            jar_path = os.path.join(stanford_pos_path, 'stanford-postagger.jar')
            
            # Validate that files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Stanford language model not found at: {model_path}\n"
                    f"Please check that:\n"
                    f"  1. stanford_pos_path is correct: {stanford_pos_path}\n"
                    f"  2. stanford_language_path is correct: {stanford_language_path}\n"
                    f"  3. The Stanford POS tagger has been downloaded from:\n"
                    f"     https://nlp.stanford.edu/software/tagger.shtml#Download"
                )
            
            if not os.path.exists(jar_path):
                raise FileNotFoundError(
                    f"Stanford POS tagger JAR not found at: {jar_path}\n"
                    f"Please check that stanford_pos_path is correct: {stanford_pos_path}\n"
                    f"The JAR file should be in the root of the Stanford directory."
                )
            
            # Create Stanford tagger
            print(f"Initializing Stanford POS tagger...")
            print(f"  Model: {model_path}")
            print(f"  JAR: {jar_path}")
            
            try:
                stanford_tagger = StanfordPOSTagger(model_path, jar_path)
                print("Stanford tagger initialized successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Stanford POS tagger: {str(e)}\n"
                    f"This may be due to:\n"
                    f"  1. Java not being installed or not in PATH\n"
                    f"  2. Incorrect JAR or model paths\n"
                    f"  3. Incompatible Stanford tagger version"
                )
        
        elif tagger_type == 'spacy':
            # Initialize spaCy tagger
            print("Initializing spaCy POS tagger...")
            spacy_nlp = initialize_spacy_tagger(spacy_model)
        
        else:
            raise ValueError(
                f"Invalid tagger_type: '{tagger_type}'\n"
                f"Must be either 'stanford' or 'spacy'"
            )

    # CRITICAL FIX: Store token/lemma objects separately before converting to strings
    # This allows us to apply POS tagging to the actual list objects
    df['token_obj'] = df['token'].copy()
    df['lemma_obj'] = df['lemma'].copy()
    
    # Convert token and lemma columns to string representations
    # This is REQUIRED for compatibility with alignment analysis scripts
    print("Converting tokens and lemmas to string representations...")
    df['token'] = df['token'].apply(str)
    df['lemma'] = df['lemma'].apply(str)
    
    # Apply NLTK POS tagging (ALWAYS included)
    print("Applying NLTK POS tagging...")
    df['tagged_token'] = df['token_obj'].apply(
        lambda x: str(nltk.pos_tag(x)) if isinstance(x, list) and x else str([])
    )
    df['tagged_lemma'] = df['lemma_obj'].apply(
        lambda x: str(nltk.pos_tag(x)) if isinstance(x, list) and x else str([])
    )
    print("NLTK POS tagging complete")
    
    # If desired, add second tagger (Stanford or spaCy) with appropriate column names
    if add_additional_tags:
        if tagger_type == 'stanford':
            print(f"Applying Stanford POS tagging with batch processing...")
            print(f"  Batch size: {stanford_batch_size} utterances")
            print(f"  Total utterances: {len(df)}")
            
            try:
                # Process tokens and lemmas in batches
                df['tagged_stan_token'] = batch_stanford_tag(
                    df['token_obj'].tolist(),
                    stanford_tagger,
                    batch_size=stanford_batch_size
                )
                df['tagged_stan_lemma'] = batch_stanford_tag(
                    df['lemma_obj'].tolist(),
                    stanford_tagger,
                    batch_size=stanford_batch_size
                )
                print("Stanford POS tagging complete")
            except Exception as e:
                print(f"WARNING: Stanford POS tagging failed: {str(e)}")
                print("Continuing without Stanford tags...")
                df['tagged_stan_token'] = str([])
                df['tagged_stan_lemma'] = str([])
        
        elif tagger_type == 'spacy':
            print(f"Applying spaCy POS tagging to {len(df)} utterances...")
            
            try:
                # Use tqdm for progress indication
                tqdm.pandas(desc="spaCy tagging tokens")
                df['tagged_spacy_token'] = df['token_obj'].progress_apply(
                    lambda x: str(spacy_tag_tokens(x, spacy_nlp)) if isinstance(x, list) and x else str([])
                )
                
                tqdm.pandas(desc="spaCy tagging lemmas")
                df['tagged_spacy_lemma'] = df['lemma_obj'].progress_apply(
                    lambda x: str(spacy_tag_tokens(x, spacy_nlp)) if isinstance(x, list) and x else str([])
                )
                print("spaCy POS tagging complete")
            except Exception as e:
                print(f"WARNING: spaCy POS tagging failed: {str(e)}")
                print("Continuing without spaCy tags...")
                df['tagged_spacy_token'] = str([])
                df['tagged_spacy_lemma'] = str([])
    
    # Add filename
    df['file'] = filename
    
    # Drop temporary object columns
    df = df.drop(['token_obj', 'lemma_obj'], axis=1)
    
    # Ensure proper column order for compatibility with alignment scripts
    column_order = [
        'participant', 'content', 'token', 'lemma',
        'tagged_token', 'tagged_lemma'
    ]
    if add_additional_tags:
        if tagger_type == 'stanford':
            column_order.extend(['tagged_stan_token', 'tagged_stan_lemma'])
        elif tagger_type == 'spacy':
            column_order.extend(['tagged_spacy_token', 'tagged_spacy_lemma'])
    column_order.append('file')
    
    # Only include columns that exist (in case tagging was skipped due to errors)
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df


def prepare_transcripts(input_files,
                        output_file_directory,
                        run_spell_check=True,
                        training_dictionary=None,
                        minwords=2,
                        use_filler_list=None,
                        filler_regex_and_list=False,
                        add_additional_tags=False,
                        tagger_type='stanford',
                        stanford_pos_path=None,
                        stanford_language_path=None,
                        stanford_batch_size=50,
                        spacy_model='en_core_web_sm',
                        input_as_directory=True,
                        save_concatenated_dataframe=True):
    """
    Prepare transcript files for linguistic alignment analysis.

    Given individual .txt files of conversations, returns completely prepared
    dataframes ready for ALIGN analysis, including: text cleaning, turn merging,
    spell-checking, tokenization, lemmatization, and POS tagging.

    Parameters
    ----------
    input_files : str or list of str
        Directory containing input files (if input_as_directory=True) or
        list of file paths to process
    output_file_directory : str
        Directory where processed files will be saved
    run_spell_check : bool, optional
        Whether to run spell-checking algorithm (default: True)
    training_dictionary : str, optional
        Path to custom spell-check training corpus. If None, uses the 
        included Project Gutenberg corpus (default: None)
    minwords : int, optional
        Minimum number of words per turn. Turns with fewer words are removed.
        Should be >= max_ngram used in later alignment analysis (default: 2)
    use_filler_list : list of str, optional
        Custom list of filler words to remove. If None, uses regex to remove
        common fillers (default: None)
    filler_regex_and_list : bool, optional
        If True, use both regex and custom filler list (default: False)
    add_additional_tags : bool, optional
        Whether to add a second set of POS tags using an alternative tagger
        (default: False)
    tagger_type : str, optional
        Which tagger to use for additional tags:
        - 'stanford': Stanford CoreNLP → creates tagged_stan_token, tagged_stan_lemma
        - 'spacy': spaCy → creates tagged_spacy_token, tagged_spacy_lemma
        (default: 'stanford' for backward compatibility)
    stanford_pos_path : str, optional
        Path to Stanford POS tagger directory (required if tagger_type='stanford')
    stanford_language_path : str, optional
        Relative path to Stanford language model (required if tagger_type='stanford')
        Example: 'models/english-left3words-distsim.tagger'
    stanford_batch_size : int, optional
        Number of utterances to process per batch for Stanford tagging.
        Larger values are faster but use more memory (default: 50)
    spacy_model : str, optional
        spaCy model to use (default: 'en_core_web_sm')
    input_as_directory : bool, optional
        Whether input_files is a directory path (True) or list of files (False)
        (default: True)
    save_concatenated_dataframe : bool, optional
        Whether to save a single concatenated file of all processed transcripts
        in addition to individual files (default: True)

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe of all processed transcripts, ready for
        alignment analysis

    Examples
    --------
    # Basic usage (NLTK only, fastest)
    >>> results = prepare_transcripts(
    ...     input_files="./raw_transcripts",
    ...     output_file_directory="./preprocessed"
    ... )
    
    # With spaCy tagging (recommended for speed)
    >>> results = prepare_transcripts(
    ...     input_files="./raw_transcripts",
    ...     output_file_directory="./preprocessed",
    ...     add_additional_tags=True,
    ...     tagger_type='spacy'
    ... )
    
    # With Stanford tagging (slowest, highest accuracy)
    >>> results = prepare_transcripts(
    ...     input_files="./raw_transcripts",
    ...     output_file_directory="./preprocessed",
    ...     add_additional_tags=True,
    ...     tagger_type='stanford',
    ...     stanford_pos_path="/path/to/stanford-postagger-full-2020-11-17/",
    ...     stanford_language_path="models/english-left3words-distsim.tagger"
    ... )
    
    Notes
    -----
    Input files must be tab-delimited with 'participant' and 'content' columns.
    Output files will have the same format with additional columns for tokens,
    lemmas, and POS tags, all stored as string representations compatible with
    ast.literal_eval() for use in alignment analysis.
    
    Tagging Speed Comparison (per 10,000 words):
    - NLTK only: ~1 second
    - NLTK + spaCy: ~1.3 seconds  
    - NLTK + Stanford (with batching): ~20-40 seconds
    """
    
    ensure_nltk_resources()

    # Create output directory if it doesn't exist
    os.makedirs(output_file_directory, exist_ok=True)
    
    # Initialize spell-checking if requested
    if run_spell_check:
        print("Initializing spell-checking model...")
        
        def train(features):
            model = defaultdict(lambda: 1)
            for f in features:
                model[f] += 1
            return model

        # Use provided dictionary or default Gutenberg corpus
        if training_dictionary is None:
            module_path = os.path.dirname(os.path.abspath(__file__))
            training_dictionary = os.path.join(module_path, 'data/gutenberg.txt')
            
            if not os.path.exists(training_dictionary):
                print(f"Warning: Default training dictionary not found at {training_dictionary}")
                print("Proceeding without spell-checking...")
                run_spell_check = False
            else:
                print(f"Using training dictionary: {training_dictionary}")
        
        if run_spell_check:
            nwords = train(re.findall('[a-z]+', open(training_dictionary).read().lower()))
            print(f"Spell-checking model trained with {len(nwords)} words")

    # Get list of files to process
    if not input_as_directory:
        file_list = glob.glob(input_files)
    else:
        file_list = glob.glob(os.path.join(input_files, "*.txt"))
    
    if not file_list:
        raise ValueError(
            f"No .txt files found in: {input_files}\n"
            f"Please check that:\n"
            f"  1. The path is correct\n"
            f"  2. The directory contains .txt files\n"
            f"  3. input_as_directory is set correctly"
        )
    
    print(f"\nFound {len(file_list)} files to process")
    print(f"Output directory: {output_file_directory}\n")

    # Process each file
    tmpfiles = []
    
    for fileName in file_list:
        try:
            print(f"{'='*60}")
            print(f"Processing: {os.path.basename(fileName)}")
            print(f"{'='*60}")
            
            # Read file
            dataframe = pd.read_csv(fileName, sep='\t', encoding='utf-8')
            
            # Validate input format
            required_cols = ['participant', 'content']
            missing_cols = [col for col in required_cols if col not in dataframe.columns]
            if missing_cols:
                print(f"ERROR: File missing required columns: {missing_cols}")
                print(f"Skipping file: {fileName}")
                continue
            
            # Clean up, merge, and process
            print("  1. Cleaning text...")
            dataframe = InitialCleanup(
                dataframe,
                minwords=minwords,
                use_filler_list=use_filler_list,
                filler_regex_and_list=filler_regex_and_list
            )
            
            print("  2. Merging adjacent turns...")
            dataframe = AdjacentMerge(dataframe)
            
            # Tokenize and lemmatize
            print("  3. Tokenizing...")
            if run_spell_check:
                print("     (with spell-checking)")
                dataframe['token'] = dataframe['content'].apply(TokenizeSpell, args=(nwords,))
            else:
                dataframe['token'] = dataframe['content'].apply(Tokenize)
            
            print("  4. Lemmatizing...")
            dataframe['lemma'] = dataframe['token'].apply(Lemmatize)

            # Apply POS tagging
            print("  5. Applying POS tagging...")
            dataframe = ApplyPOSTagging(
                dataframe,
                filename=os.path.basename(fileName),
                add_additional_tags=add_additional_tags,
                tagger_type=tagger_type,
                stanford_pos_path=stanford_pos_path,
                stanford_language_path=stanford_language_path,
                stanford_batch_size=stanford_batch_size,
                spacy_model=spacy_model
            )

            # Save processed file
            conversation_file = os.path.join(output_file_directory, os.path.basename(fileName))
            dataframe.to_csv(conversation_file, encoding='utf-8', index=False, sep='\t')
            print(f"  6. Saved: {os.path.basename(conversation_file)}")
            print(f"     Rows: {len(dataframe)}")
            
            tmpfiles.append(dataframe)
            
        except Exception as e:
            print(f"ERROR processing {fileName}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this file and continuing...")
            continue

    # Concatenate all results
    if not tmpfiles:
        raise ValueError(
            "No files were successfully processed. Please check:\n"
            "  1. Input file format (tab-delimited with 'participant' and 'content' columns)\n"
            "  2. File encoding (should be UTF-8)\n"
            "  3. Error messages above for specific issues"
        )
    
    prepped_df = pd.concat(tmpfiles, ignore_index=True)
    
    # Save concatenated dataframe
    if save_concatenated_dataframe:
        concatenated_file = os.path.join(output_file_directory, 'align_concatenated_dataframe.txt')
        prepped_df.to_csv(concatenated_file, encoding='utf-8', index=False, sep='\t')
        print(f"\n{'='*60}")
        print(f"Saved concatenated dataframe: {os.path.basename(concatenated_file)}")
        print(f"Total rows: {len(prepped_df)}")
        print(f"{'='*60}\n")

    print("Processing complete!")
    print(f"\nSummary:")
    print(f"  - Files processed: {len(tmpfiles)}")
    print(f"  - Total utterances: {len(prepped_df)}")
    print(f"  - Output directory: {output_file_directory}")
    
    return prepped_df