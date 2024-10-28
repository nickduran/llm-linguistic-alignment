import os,re,math,csv,string,random,logging,glob,itertools,operator,sys
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict, OrderedDict
from itertools import chain, combinations

import pandas as pd
import numpy as np
import scipy
from scipy import spatial

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from nltk.util import ngrams

import gensim
from gensim.models import word2vec

def ngram_pos(sequence1,sequence2,ngramsize=2,
                   ignore_duplicates=True):
    """
    Remove mimicked lexical sequences from two interlocutors'
    sequences and return a dictionary of counts of ngrams
    of the desired size for each sequence.

    By default, consider bigrams. If desired, this may be
    changed by setting `ngramsize` to the appropriate
    value.

    By default, ignore duplicate lexical n-grams when
    processing these sequences. If desired, this may
    be changed with `ignore_duplicates=False`.
    """

    # remove duplicates and recreate sequences
    sequence1 = set(ngrams(sequence1,ngramsize))
    sequence2 = set(ngrams(sequence2,ngramsize))

    # if desired, remove duplicates from sequences
    if ignore_duplicates:
        new_sequence1 = [tuple([''.join(pair[1]) for pair in tup]) for tup in list(sequence1 - sequence2)]
        new_sequence2 = [tuple([''.join(pair[1]) for pair in tup]) for tup in list(sequence2 - sequence1)]
    else:
        new_sequence1 = [tuple([''.join(pair[1]) for pair in tup]) for tup in sequence1]
        new_sequence2 = [tuple([''.join(pair[1]) for pair in tup]) for tup in sequence2]

    # return counters
    return Counter(new_sequence1), Counter(new_sequence2)


def ngram_lexical(sequence1,sequence2,ngramsize=2):
    """
    Create ngrams of the desired size for each of two
    interlocutors' sequences and return a dictionary
    of counts of ngrams for each sequence.

    By default, consider bigrams. If desired, this may be
    changed by setting `ngramsize` to the appropriate
    value.
    """

    # generate ngrams
    sequence1 = list(ngrams(sequence1,ngramsize))
    sequence2 = list(ngrams(sequence2,ngramsize))

    # join for counters
    new_sequence1 = [' '.join(pair) for pair in sequence1]
    new_sequence2 = [' '.join(pair) for pair in sequence2]

    # return counters
    return Counter(new_sequence1), Counter(new_sequence2)


def get_cosine(vec1, vec2):
    """
    Derive cosine similarity metric, standard measure.
    Adapted from <https://stackoverflow.com/a/33129724>.
    """

    # NOTE: results identical to sklearn class `cosine_similarity` EXCEPT sklearn removes single letter words, the ALIGN method does not
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x]**2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def build_composite_semantic_vector(lemma_seq,vocablist,highDimModel):
    """
    Function for producing vocablist and model is called in the main loop
    """
    
    # vocablist are all unique lemmas except those removed if only occurred once or user-specified setting of frequency threshold
    filter_lemma_seq = [word for word in lemma_seq if word in vocablist]
    # build composite vector via simple sum of vectors
    getComposite = [0] * len(highDimModel[vocablist[1]])
    for w1 in filter_lemma_seq:
        if w1 in highDimModel:
            semvector = highDimModel[w1]
            getComposite = getComposite + semvector
    return getComposite


def BuildSemanticModel(semantic_model_input_file,
                        pretrained_input_file,
                        output_file_directory,
                        use_pretrained_vectors=True,
                        high_sd_cutoff=3,
                        low_n_cutoff=1,
                        save_vocab_freqs=False):

    """
    Given an input file produced by the ALIGN Phase 1 functions,
    build a semantic model from all transcripts in all conversations
    in target corpus after removing high- and low-frequency words.
    High-frequency words are determined by a user-defined number of
    SDs over the mean (by default, `high_sd_cutoff=3`). Low-frequency
    words must appear over a specified number of raw occurrences
    (by default, `low_n_cutoff=1`).

    Frequency cutoffs can be removed by `high_sd_cutoff=None` and/or
    `low_n_cutoff=0`.
    
    Also optionally saves to `output_file_directory` a .txt list of all 
    unique words and their frequency count if `save_vocab_freqs=True`
    """

    # build vocabulary list from transcripts
    data1 = pd.read_csv(semantic_model_input_file, sep='\t', encoding='utf-8')

    # get frequency count of all included words
    all_sentences = [re.sub('[^\w\s]+','',str(row)).split(' ') for row in list(data1['lemma'])]
    all_words = list([a for b in all_sentences for a in b])
    frequency = defaultdict(int)
    for word in all_words:
        frequency[word] += 1

    # only include words whose length is greater than 1
    frequency = {word: freq for word, freq in frequency.items() if len(word) > 1}

    # NOTE: experimental feature for 0.0.12, print out list of all unique words with frequency count    
    if save_vocab_freqs:
        vocabfreq_df = pd.DataFrame(list(frequency.items()), columns=["word", "count"]).sort_values(by=['count'], ascending=False)
        vocabfreq_file = os.path.join(output_file_directory,'../vocabfreqs.txt')
        vocabfreq_df.to_csv(vocabfreq_file,
                    encoding='utf-8',index=False, sep='\t')
    
    # only include words that occur more frequently than our cutoff (defined in occurrences)
    frequency = {word: freq for word, freq in frequency.items() if freq > low_n_cutoff}

    # if desired, remove high-frequency words (over user-defined SDs above mean)
    if high_sd_cutoff is None:
        contentWords = [word for word in list(frequency.keys())]
    else:
        getOut = np.mean(list(frequency.values()))+(np.std(list(frequency.values()))*(high_sd_cutoff))
        contentWords = list({word: freq for word, freq in frequency.items() if freq < getOut}.keys())

    # decide whether to build semantic model from scratch or load in pretrained vectors
    if not use_pretrained_vectors:
        keepSentences = [[word for word in row if word in contentWords] for row in all_sentences]
        semantic_model = word2vec.Word2Vec(all_sentences, min_count=low_n_cutoff)
    else:
        if pretrained_input_file is None:
            raise ValueError('Error! Specify path to pretrained vector file using the `pretrained_input_file` argument.')
        else:
            semantic_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_input_file, binary=True)

    # return all the content words and the trained word vectors
    word_to_vector = {}
    for word, index in semantic_model.key_to_index.items():
        word_to_vector[word] = semantic_model.vectors[index, :]
    return contentWords, word_to_vector

def LexicalPOSAlignment(tok1,lem1,penn_tok1,penn_lem1,
                             tok2,lem2,penn_tok2,penn_lem2,
                             stan_tok1=None,stan_lem1=None,
                             stan_tok2=None,stan_lem2=None,
                             maxngram=2,
                             ignore_duplicates=True,
                             add_stanford_tags=False):

    """
    Derive lexical and part-of-speech alignment scores
    between interlocutors (suffix `1` and `2` in arguments
    passed to function).

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True` and by providing appropriate
    values for `stan_tok1`, `stan_lem1`, `stan_tok2`, and
    `stan_lem2`.

    By default, consider only bigram when calculating
    similarity. If desired, this window may be expanded
    by changing the `maxngram` argument value.

    By default, remove exact duplicates when calculating
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # create empty dictionaries for syntactic similarity
    syntax_penn_tok = {}
    syntax_penn_lem = {}

    # if desired, generate Stanford-based scores
    if add_stanford_tags:
        syntax_stan_tok = {}
        syntax_stan_lem = {}

    # create empty dictionaries for lexical similarity
    lexical_tok = {}
    lexical_lem = {}

    # cycle through all desired ngram lengths
    for ngram in range(1,maxngram+1):

        # calculate similarity for lexical ngrams (tokens and lemmas)
        [vectorT1, vectorT2] = ngram_lexical(tok1,tok2,ngramsize=ngram)
        [vectorL1, vectorL2] = ngram_lexical(lem1,lem2,ngramsize=ngram)
        lexical_tok['lexical_tok{0}'.format(ngram)] = get_cosine(vectorT1,vectorT2)
        lexical_lem['lexical_lem{0}'.format(ngram)] = get_cosine(vectorL1, vectorL2)

        # calculate similarity for Penn POS ngrams (tokens)
        [vector_penn_tok1, vector_penn_tok2] = ngram_pos(penn_tok1,penn_tok2,
                                                ngramsize=ngram,
                                                ignore_duplicates=ignore_duplicates)
        syntax_penn_tok['syntax_penn_tok{0}'.format(ngram)] = get_cosine(vector_penn_tok1,
                                                                                            vector_penn_tok2)
        # calculate similarity for Penn POS ngrams (lemmas)
        [vector_penn_lem1, vector_penn_lem2] = ngram_pos(penn_lem1,penn_lem2,
                                                              ngramsize=ngram,
                                                              ignore_duplicates=ignore_duplicates)
        syntax_penn_lem['syntax_penn_lem{0}'.format(ngram)] = get_cosine(vector_penn_lem1,
                                                                                            vector_penn_lem2)

        # if desired, also calculate using Stanford POS
        if add_stanford_tags:

            # calculate similarity for Stanford POS ngrams (tokens)
            [vector_stan_tok1, vector_stan_tok2] = ngram_pos(stan_tok1,stan_tok2,
                                                                  ngramsize=ngram,
                                                                  ignore_duplicates=ignore_duplicates)
            syntax_stan_tok['syntax_stan_tok{0}'.format(ngram)] = get_cosine(vector_stan_tok1,
                                                                                                vector_stan_tok2)

            # calculate similarity for Stanford POS ngrams (lemmas)
            [vector_stan_lem1, vector_stan_lem2] = ngram_pos(stan_lem1,stan_lem2,
                                                                  ngramsize=ngram,
                                                                  ignore_duplicates=ignore_duplicates)
            syntax_stan_lem['syntax_stan_lem{0}'.format(ngram)] = get_cosine(vector_stan_lem1,
                                                                                                vector_stan_lem2)

    # return requested information
    if add_stanford_tags:
        dictionaries_list = [syntax_penn_tok, syntax_penn_lem,
                             syntax_stan_tok, syntax_stan_lem,
                             lexical_tok, lexical_lem]
    else:
        dictionaries_list = [syntax_penn_tok, syntax_penn_lem,
                             lexical_tok, lexical_lem]

    return dictionaries_list


def conceptualAlignment(lem1, lem2, vocablist, highDimModel):

    """
    Calculate conceptual alignment scores from list of lemmas
    from between two interocutors (suffix `1` and `2` in arguments
    passed to function) using `word2vec`.
    """

    # aggregate composite high-dimensional vectors of all words in utterance
    W2Vec1 = build_composite_semantic_vector(lem1,vocablist,highDimModel)
    W2Vec2 = build_composite_semantic_vector(lem2,vocablist,highDimModel)

    # return cosine distance alignment score; checks whether one of the vectors is composed of all zeros (occasionaly occurs when words in utterance were removed from master vocablist because of low or high frequency issues)
    if all(v == 0 for v in W2Vec1) | all(v == 0 for v in W2Vec2):
        return 0.0   
    else:
        return 1 - spatial.distance.cosine(W2Vec1, W2Vec2)

def returnMultilevelAlignment(cond_info,
                                   partnerA,tok1,lem1,penn_tok1,penn_lem1,
                                   partnerB,tok2,lem2,penn_tok2,penn_lem2,
                                   vocablist, highDimModel,
                                   stan_tok1=None,stan_lem1=None,
                                   stan_tok2=None,stan_lem2=None,
                                   add_stanford_tags=False,
                                   maxngram=2,
                                   ignore_duplicates=True):

    """
    Calculate lexical, syntactic, and conceptual alignment
    between a pair of turns by individual interlocutors
    (suffix `1` and `2` in arguments passed to function),
    including leading/following comparison directionality.

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True` and by providing appropriate
    values for `stan_tok1`, `stan_lem1`, `stan_tok2`, and
    `stan_lem2`.

    By default, consider only bigrams when calculating
    similarity. If desired, this window may be expanded
    by changing the `maxngram` argument value.

    By default, remove exact duplicates when calculating
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # create empty dictionaries
    partner_direction = {}
    condition_info = {}
    cosine_semanticL = {}
    utterance_length1 = {}
    utterance_length2 = {}

    # calculate lexical and syntactic alignment
    dictionaries_list = LexicalPOSAlignment(tok1=tok1,lem1=lem1,
                                                 penn_tok1=penn_tok1,penn_lem1=penn_lem1,
                                                 tok2=tok2,lem2=lem2,
                                                 penn_tok2=penn_tok2,penn_lem2=penn_lem2,
                                                 stan_tok1=stan_tok1,stan_lem1=stan_lem1,
                                                 stan_tok2=stan_tok2,stan_lem2=stan_lem2,
                                                 maxngram=maxngram,
                                                 ignore_duplicates=ignore_duplicates,
                                                 add_stanford_tags=add_stanford_tags)

    # calculate conceptual alignment
    cosine_semanticL['cosine_semanticL'] = conceptualAlignment(lem1,lem2,vocablist,highDimModel)
    dictionaries_list.append(cosine_semanticL.copy())

    # determine directionality of leading/following comparison;  Note: Partner B is the lagged partner, thus, B is following A
    partner_direction['partner_direction'] = str(partnerA) + ">" + str(partnerB)
    dictionaries_list.append(partner_direction.copy())

    # add number of tokens in each utterance
    utterance_length1['utterance_length1'] = len(tok1)
    dictionaries_list.append(utterance_length1.copy())

    utterance_length2['utterance_length2'] = len(tok2)
    dictionaries_list.append(utterance_length2.copy())

    # add condition information
    condition_info['condition_info'] = cond_info
    dictionaries_list.append(condition_info.copy())

    # return alignment scores
    return dictionaries_list


def TurnByTurnAnalysis(dataframe,
                            vocablist,
                            highDimModel,
                            delay=1,
                            maxngram=2,
                            add_stanford_tags=False,
                            ignore_duplicates=True):

    """
    Calculate lexical, syntactic, and conceptual alignment
    between interlocutors over an entire conversation.
    Automatically detect individual speakers by unique
    speaker codes.

    By default, compare only adjacent turns. If desired,
    the comparison distance may be changed by increasing
    the `delay` argument.

    By default, include maximum n-gram comparison of 2. If
    desired, this may be changed by passing the appropriate
    value to the the `maxngram` argument.

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True`.

    By default, remove exact duplicates when calculating POS
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # if we don't want the Stanford tagger data, set defaults
    if not add_stanford_tags:
        stan_tok1=None
        stan_lem1=None
        stan_tok2=None
        stan_lem2=None

    # prepare the data to the appropriate type
    dataframe['token'] = dataframe['token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['lemma'] = dataframe['lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['tagged_token'] = dataframe['tagged_token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['tagged_token'] = dataframe['tagged_token'].apply(lambda x: list(zip(x[0::2],x[1::2]))) # thanks to https://stackoverflow.com/a/4647086
    dataframe['tagged_lemma'] = dataframe['tagged_lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['tagged_lemma'] = dataframe['tagged_lemma'].apply(lambda x: list(zip(x[0::2],x[1::2]))) # thanks to https://stackoverflow.com/a/4647086

    # if desired, prepare the Stanford tagger data
    if add_stanford_tags:
        dataframe['tagged_stan_token'] = dataframe['tagged_stan_token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
        dataframe['tagged_stan_token'] = dataframe['tagged_stan_token'].apply(lambda x: list(zip(x[0::2],x[1::2]))) # thanks to https://stackoverflow.com/a/4647086
        dataframe['tagged_stan_lemma'] = dataframe['tagged_stan_lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
        dataframe['tagged_stan_lemma'] = dataframe['tagged_stan_lemma'].apply(lambda x: list(zip(x[0::2],x[1::2]))) # thanks to https://stackoverflow.com/a/4647086

    # create lagged version of the dataframe
    df_original = dataframe.drop(dataframe.tail(delay).index,inplace=False)
    df_lagged = dataframe.shift(-delay).drop(dataframe.tail(delay).index,inplace=False)

    # cycle through each pair of turns
    # aggregated_df = pd.DataFrame()
    tmpfiles = list()
    
    for i in range(0,df_original.shape[0]):

        # identify the condition for this dataframe
        cond_info = dataframe['file'].unique()
        if len(cond_info)==1:
            cond_info = str(cond_info[0])

        # break and flag error if we have more than 1 condition per dataframe
        else:
            raise ValueError('Error! Dataframe contains multiple conditions. Split dataframe into multiple dataframes, one per condition: '+cond_info)

        # grab all of first participant's data
        first_row = df_original.iloc[i]
        first_partner = first_row['participant']
        tok1=first_row['token']
        lem1=first_row['lemma']
        penn_tok1=first_row['tagged_token']
        penn_lem1=first_row['tagged_lemma']

        # grab all of lagged participant's data
        lagged_row = df_lagged.iloc[i]
        lagged_partner = lagged_row['participant']
        tok2=lagged_row['token']
        lem2=lagged_row['lemma']
        penn_tok2=lagged_row['tagged_token']
        penn_lem2=lagged_row['tagged_lemma']

        # if desired, grab the Stanford tagger data for both participants
        if add_stanford_tags:
            stan_tok1=first_row['tagged_stan_token']
            stan_lem1=first_row['tagged_stan_lemma']
            stan_tok2=lagged_row['tagged_stan_token']
            stan_lem2=lagged_row['tagged_stan_lemma']

        # process multilevel alignment
        dictionaries_list=returnMultilevelAlignment(cond_info=cond_info,
                                                         partnerA=first_partner,
                                                         tok1=tok1,lem1=lem1,
                                                         penn_tok1=penn_tok1,penn_lem1=penn_lem1,
                                                         partnerB=lagged_partner,
                                                         tok2=tok2,lem2=lem2,
                                                         penn_tok2=penn_tok2,penn_lem2=penn_lem2,
                                                         vocablist=vocablist,
                                                         highDimModel=highDimModel,
                                                         stan_tok1=stan_tok1,stan_lem1=stan_lem1,
                                                         stan_tok2=stan_tok2,stan_lem2=stan_lem2,
                                                         maxngram = maxngram,
                                                         ignore_duplicates = ignore_duplicates,
                                                         add_stanford_tags = add_stanford_tags)

        # sort columns so they are in order, append data to existing structures
        next_df_line = pd.DataFrame.from_dict(OrderedDict(k for num, i in enumerate(d for d in dictionaries_list) for k in sorted(i.items())),
                               orient='index').transpose()
            
        # aggregated_df = aggregated_df.append(next_df_line) ## problematic. appending a dataframe to a dataframe. 
        tmpfiles.append(next_df_line)    
    
    # reformat turn information and add index
    aggregated_df = pd.concat(tmpfiles)
    aggregated_df = aggregated_df.reset_index(drop=True).reset_index().rename(columns={"index":"time"})

    # give us our finished dataframe
    return aggregated_df

def calculate_alignment(input_files,
                        output_file_directory,
                        semantic_model_input_file,
                        pretrained_input_file,
                        high_sd_cutoff=3,
                        low_n_cutoff=1,
                        delay=1,
                        maxngram=2,
                        use_pretrained_vectors=True,
                        ignore_duplicates=True,
                        add_stanford_tags=False,
                        input_as_directory=True,
                        save_vocab_freqs=False):

    """
    Calculate lexical, syntactic, and conceptual alignment between speakers.

    Given a directory of individual .txt files and the
    vocabulary list that have been generated by the `prepare_transcripts`
    preparation stage, return multi-level alignment
    scores with turn-by-turn and conversation-level metrics.

    Parameters
    ----------

    input_files : str (directory name) or list of str (file names)
        Cleaned files to be analyzed. Behavior governed by `input_as_directory`
        parameter as well.

    output_file_directory : str
        Name of directory where output for individual conversations will be
        saved.

    semantic_model_input_file : str
        Name of file to be used for creating the semantic model. A compatible
        file will be saved as an output of `prepare_transcripts()`.

    pretrained_input_file : str or None
        If using a pretrained vector to create the semantic model, use
        name of model here. If not, use None. Behavior governed by
        `use_pretrained_vectors` parameter as well.

    high_sd_cutoff : int, optional (default: 3)
        High-frequency cutoff (in SD over the mean) for lexical items
        when creating the semantic model.

    low_n_cutoff : int, optional (default: 1)
        Low-frequency cutoff (in raw frequency) for lexical items when
        creating the semantic models. Items with frequency less than or
        equal to the number provided here will be removed. To remove the
        low-frequency cutoff, set to 0.

    delay : int, optional (default: 1)
        Delay (or lag) at which to calculate similarity. A lag of 1 (default)
        considers only adjacent turns.

    maxngram : int, optional (default: 2)
        Maximum n-gram size for calculations. Similarity scores for n-grams
        from unigrams to the maximum size specified here will be calculated.

    use_pretrained_vectors : boolean, optional (default: True)
        Specify whether to use a pretrained gensim model for word2vec
        analysis (True) or to construct a new model from the provided corpus
        (False). If True, the file name of a valid model must be
        provided to the `pretrained_input_file` parameter.

    ignore_duplicates : boolean, optional (default: True)
        Specify whether to remove exact duplicates when calculating
        part-of-speech similarity scores (True) or to retain perfectly
        mimicked lexical items for POS similarity calculation (False).

    add_stanford_tags : boolean, optional (default: False)
        Specify whether to return part-of-speech similarity scores based on
        Stanford POS tagger in addition to the Penn POS tagger (True) or to
        return only POS similarity scores from the Penn tagger (False). (Note:
        Including Stanford POS tags will lead to a significant increase in
        processing time.)

    input_as_directory : boolean, optional (default: True)
        Specify whether the value passed to `input_files` parameter should
        be read as a directory (True) or a list of files to be processed
        (False).
        
    save_vocab_freqs: boolean, optional (default: False)
        Specify whether to save a .txt file to `output_file_directory` that 
        contains list of all unique words in corpus with frequency counts 

    Returns
    -------

    real_final_turn_df : Pandas DataFrame
        A dataframe of lexical, syntactic, and conceptual alignment scores
        between turns at specified delay. `NaN` values will be returned for
        turns in which the speaker only produced words that were removed
        from the corpus (e.g., too rare or too common words) or words that were
        present in the corpus but not in the semantic model.

    """

    # grab the files in the list
    if not input_as_directory:
        file_list = glob.glob(input_files)
    else:
        file_list = glob.glob(input_files+"/*.txt")

    # build the semantic model to be used for all conversations
    [vocablist, highDimModel] = BuildSemanticModel(semantic_model_input_file=semantic_model_input_file,
                                                       pretrained_input_file=pretrained_input_file,
                                                       use_pretrained_vectors=use_pretrained_vectors,
                                                       high_sd_cutoff=high_sd_cutoff,
                                                       low_n_cutoff=low_n_cutoff,
                                                       save_vocab_freqs=save_vocab_freqs,
                                                       output_file_directory=output_file_directory)

    # create containers for alignment values
    tempT2T = list()
    tempC2C = list()

    # cycle through each prepared file
    for fileName in file_list:

        # process the file if it's got a valid conversation
        dataframe=pd.read_csv(fileName, sep='\t',encoding='utf-8')
        if len(dataframe) > 1:

            # let us know which filename we're processing
            print(("Processing: "+fileName))

            # calculate turn-by-turn alignment scores
            xT2T=TurnByTurnAnalysis(dataframe=dataframe,
                                         delay=delay,
                                         maxngram=maxngram,
                                         vocablist=vocablist,
                                         highDimModel=highDimModel,
                                         add_stanford_tags=add_stanford_tags,
                                         ignore_duplicates=ignore_duplicates)
            tempT2T.append(xT2T)

        # if it's invalid, let us know
        else:
            print(("Invalid file: "+fileName))

    # update final dataframes
    AlignmentT2T = pd.concat(tempT2T)
    real_final_turn_df = AlignmentT2T.reset_index(drop=True)
    AlignmentC2C = pd.concat(tempC2C)
    real_final_convo_df = AlignmentC2C.reset_index(drop=True)

    # export the final files
    real_final_turn_df.to_csv(output_file_directory+"AlignmentT2T.txt",
                      encoding='utf-8', index=False, sep='\t')

    # display the info, too
    return real_final_turn_df, real_final_convo_df









                                 surrogate_file_directory,
                                 output_file_directory,
                                 semantic_model_input_file,
                                 pretrained_input_file,
                                 high_sd_cutoff=3,
                                 low_n_cutoff=1,
                                 id_separator='\-',
                                 condition_label='cond',
                                 dyad_label='dyad',
                                 all_surrogates=True,
                                 keep_original_turn_order=True,
                                 delay=1,
                                 maxngram=2,
                                 use_pretrained_vectors=True,
                                 ignore_duplicates=True,
                                 add_stanford_tags=False,
                                 input_as_directory=True,
                                 save_vocab_freqs=False):

    """
    Calculate baselines for lexical, syntactic, and conceptual
    alignment between speakers.

    Given a directory of individual .txt files and the
    vocab list that have been generated by the `prepare_transcripts`
    preparation stage, return multi-level alignment
    scores with turn-by-turn and conversation-level metrics
    for surrogate baseline conversations.

    Parameters
    ----------

    input_files : str (directory name) or list of str (file names)
        Cleaned files to be analyzed. Behavior governed by `input_as_directory`
        parameter as well.

    surrogate_file_directory : str
        Name of directory where raw surrogate data will be saved.

    output_file_directory : str
        Name of directory where output for individual surrogate
        conversations will be saved.

    semantic_model_input_file : str
        Name of file to be used for creating the semantic model. A compatible
        file will be saved as an output of `prepare_transcripts()`.

    pretrained_input_file : str or None
        If using a pretrained vector to create the semantic model, use
        name of model here. If not, use None. Behavior governed by
        `use_pretrained_vectors` parameter as well.

    high_sd_cutoff : int, optional (default: 3)
        High-frequency cutoff (in SD over the mean) for lexical items
        when creating the semantic model.

    low_n_cutoff : int, optional (default: 1)
        Low-frequency cutoff (in raw frequency) for lexical items when
        creating the semantic models. Items with frequency less than or
        equal to the number provided here will be removed. To remove the
        low-frequency cutoff, set to 0.

    id_separator : str, optional (default: '\-')
        Character separator between the dyad and condition IDs in
        original data file names.

    condition_label : str, optional (default: 'cond')
        String preceding ID for each unique condition. Anything after this
        label will be identified as a unique condition ID.

    dyad_label : str, optional (default: 'dyad')
        String preceding ID for each unique dyad. Anything after this label
        will be identified as a unique dyad ID.

    all_surrogates : boolean, optional (default: True)
        Specify whether to generate all possible surrogates across original
        dataset (True) or to generate only a subset of surrogates equal to
        the real sample size drawn randomly from all possible surrogates
        (False).

    keep_original_turn_order : boolean, optional (default: True)
        Specify whether to retain original turn ordering when pairing surrogate
        dyads (True) or to pair surrogate partners' turns in random order
        (False).

    delay : int, optional (default: 1)
        Delay (or lag) at which to calculate similarity. A lag of 1 (default)
        considers only adjacent turns.

    maxngram : int, optional (default: 2)
        Maximum n-gram size for calculations. Similarity scores for n-grams
        from unigrams to the maximum size specified here will be calculated.

    use_pretrained_vectors : boolean, optional (default: True)
        Specify whether to use a pretrained gensim model for word2vec
        analysis. If True, the file name of a valid model must be
        provided to the `pretrained_input_file` parameter.

    ignore_duplicates : boolean, optional (default: True)
        Specify whether to remove exact duplicates when calculating
        part-of-speech similarity scores. By default, ignore perfectly
        mimicked lexical items for POS similarity calculation.

    add_stanford_tags : boolean, optional (default: False)
        Specify whether to return part-of-speech similarity scores
        based on Stanford POS tagger (in addition to the Penn POS
        tagger).

    input_as_directory : boolean, optional (default: True)
        Specify whether the value passed to `input_files` parameter should
        be read as a directory or a list of files to be processed.
        
    Returns
    -------

    surrogate_final_turn_df : Pandas DataFrame
        A dataframe of lexical, syntactic, and conceptual alignment scores
        between turns at specified delay for surrogate partners. `NaN` values
        will be returned for turns in which the speaker only produced words
        that were removed from the corpus (e.g., too rare or too common words)
        or words that were present in the corpus but not in the semantic model.

    surrogate_final_convo_df : Pandas DataFrame
        A dataframe of lexical, syntactic, and conceptual alignment scores
        between surrogate partners across the entire conversation.
    """

    # grab the files in the input list
    if not input_as_directory:
        file_list = glob.glob(input_files)
    else:
        file_list = glob.glob(input_files+"/*.txt")

    # create a surrogate file list
    surrogate_file_list = GenerateSurrogate(
                            original_conversation_list=file_list,
                            surrogate_file_directory=surrogate_file_directory,
                            all_surrogates=all_surrogates,
                            id_separator=id_separator,
                            condition_label=condition_label,
                            dyad_label=dyad_label,
                            keep_original_turn_order=keep_original_turn_order)

    # build the semantic model to be used for all conversations
    [vocablist, highDimModel] = BuildSemanticModel(
                            semantic_model_input_file=semantic_model_input_file,
                            pretrained_input_file=pretrained_input_file,
                            use_pretrained_vectors=use_pretrained_vectors,
                            high_sd_cutoff=high_sd_cutoff,
                            low_n_cutoff=low_n_cutoff,
                            save_vocab_freqs=save_vocab_freqs,
                            output_file_directory=output_file_directory)                           

    # create containers for alignment values
    tempT2T = list()
    tempC2C = list()

    # cycle through the files
    for fileName in surrogate_file_list:

        # process the file if it's got a valid conversation
        dataframe=pd.read_csv(fileName, sep='\t',encoding='utf-8')
        if len(dataframe) > 1:

            # let us know which filename we're processing
            print(("Processing: "+fileName))

            # calculate turn-by-turn alignment scores
            xT2T=TurnByTurnAnalysis(dataframe=dataframe,
                                         delay=delay,
                                         maxngram=maxngram,
                                         vocablist=vocablist,
                                         highDimModel=highDimModel,
                                         add_stanford_tags = add_stanford_tags,
                                         ignore_duplicates = ignore_duplicates)
            tempT2T.append(xT2T)
                
            # calculate conversation-level alignment scores
            xC2C = ConvoByConvoAnalysis(dataframe=dataframe,
                                             maxngram = maxngram,
                                             ignore_duplicates=ignore_duplicates,
                                             add_stanford_tags = add_stanford_tags)
            tempC2C.append(xC2C)
            
        # if it's invalid, let us know
        else:
            print(("Invalid file: "+fileName))

    # update final dataframes
    AlignmentT2T = pd.concat(tempT2T)
    surrogate_final_turn_df = AlignmentT2T.reset_index(drop=True)

    # display the info, too
    return surrogate_final_turn_df, surrogate_final_convo_df