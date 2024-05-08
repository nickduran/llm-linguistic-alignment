#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 8            # number of cores 
#SBATCH -t 0-06:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q grp_cbi      # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate gpt-bert-align

python << EOF
# Built-in modules
import os
import re
import math
import glob
import pickle
import datetime

# Third-party modules
import pandas as pd
import numpy as np
import scipy
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import gensim 
import gensim.downloader as api 
from gensim.models import KeyedVectors 
import openai
from openai.embeddings_utils import get_embedding
import sentence_transformers as st

# Built-in modules from specific classes
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict, OrderedDict

def build_filtered_vocab(concat_transcripts,
                        output_file_directory,
                        high_sd_cutoff=3,
                        low_n_cutoff=1):
    
    # build vocabulary list from transcripts
    data1 = pd.read_csv(concat_transcripts, sep='\t', encoding='utf-8')

    # get frequency count of all included words (as tokens) NOTE: previous default was lemmas
    all_sentences = [re.sub('[^\w\s]+','',str(row)).split(' ') for row in list(data1['lemma'])]
    all_words = list([a for b in all_sentences for a in b])
    frequency = defaultdict(int)
    for word in all_words:
        frequency[word] += 1

    ## start filtering process

    # remove one-letter words (noise or extremely high frequency)
    frequency_filt = {word: freq for word, freq in frequency.items() if len(word) > 1}
    
    # if desired, remove words that only occur more frequently than our cutoff (defined in occurrences)
    frequency_filt = {word: freq for word, freq in frequency_filt.items() if freq > low_n_cutoff}

    # if desired, remove high-frequency words (over user-defined SDs above mean AFTER removing one-letter words [which will impact SD])
    if high_sd_cutoff is None:
        filteredWords = [word for word in list(frequency_filt.keys())]
    else:
        getOut = np.mean(list(frequency_filt.values()))+(np.std(list(frequency_filt.values()))*(high_sd_cutoff))
        # filteredWords = list({word: freq for word, freq in frequency_filt.items() if freq < getOut}.keys())
        filteredWords = {word: freq for word, freq in frequency_filt.items() if freq < getOut}

    ############ BONUS opertation: prints the frequency lists
    vocabfreq_all = pd.DataFrame(list(frequency.items()), columns=["word", "count"]).sort_values(by=['count'], ascending=False)
    vocabfreq_filt = pd.DataFrame(list(filteredWords.items()), columns=["word", "count"]).sort_values(by=['count'], ascending=False)
    
    vocabfreq_file = os.path.join(output_file_directory,'vocab_unfilt_freqs.txt')
    vocabfreq_all.to_csv(vocabfreq_file, encoding='utf-8',index=False, sep='\t')
    
    vocabfreq_filt_file = os.path.join(output_file_directory,'vocab_filt_freqs.txt')
    vocabfreq_filt.to_csv(vocabfreq_filt_file, encoding='utf-8',index=False, sep='\t')
    ############

    return list(frequency.keys()), list(filteredWords.keys())

def load_w2v_trained(pretrained_input_file):

    model = api.load(pretrained_input_file)

    return model

def process_utterance_string(tok_seq,vocablist,highDimModel):
    
    # Only consider the words that are in the vocablist after filtering for various criteria (e.g., only occur once, high frequency)
    filter_vocablist = [word for word in tok_seq if word in vocablist]
    
    # Only consider the words that are in the pre-trained model vocabulary
    filter_model = [word for word in filter_vocablist if highDimModel.has_index_for(word)]

    return filter_model
        
def process_utterance_comp_norm(tok_seq,vocablist,highDimModel):    
    
    # Only consider the words that are in the vocablist after filtering for various criteria (e.g., only occur once, high frequency)
    filter_tok_seq = [word for word in tok_seq if word in vocablist]
        
    # Retrieve the Word2Vec vectors for each word in the sentence; ignores any words not in the pre-trained model vocabulary
    word_vectors = [highDimModel[word] for word in filter_tok_seq if highDimModel.has_index_for(word)]

    # If no word vectors were retrieved, return None
    if not word_vectors:
        return None, None

    # Aggregate the Word2Vec vectors using averaging (or sum, as originally done in ALIGN)
    vector_avg = np.mean(word_vectors, axis=0)
    vector_sum = np.sum(word_vectors, axis=0)
    
    return vector_avg, vector_sum

def conceptual_alignment_w2v(sentence1, sentence2, vocablist, highDimModel):    
    # Process the sentences
    [vector1avg, vector1sum] = process_utterance_comp_norm(sentence1, vocablist, highDimModel)
    [vector2avg, vector2sum] = process_utterance_comp_norm(sentence2, vocablist, highDimModel)

    # If either vector is None (i.e., the corresponding sentence had no known words), return 0
    if vector1sum is None or vector2sum is None:
        return 0

    # Normalize the aggregated vectors
    vector1_norm = vector1sum / np.linalg.norm(vector1sum)
    vector2_norm = vector2sum / np.linalg.norm(vector2sum)

    # Calculate cosine similarity (this is equivalent to dot product between two vectors given these are normalized vectors)
    similarity = cosine_similarity([vector1_norm], [vector2_norm])
    
    return similarity[0][0]

def conceptual_alignment_GPT(content1, content2,
                           highDimModel):

    emb1 = get_embedding(content1, engine=highDimModel)
    emb2 = get_embedding(content2, engine=highDimModel)
    simGPT = cosine_similarity([emb1], [emb2])[0][0]
        
    return simGPT

def conceptual_alignment_BERT(content1, content2,
                            highDimModel):

    emb1 = highDimModel.encode(content1)
    emb2 = highDimModel.encode(content2)
    simBERT = cosine_similarity([emb1], [emb2])[0][0]
        
    return simBERT

def return_multilevel_alignment(cond_info,
                                   partnerA,tok1,lem1,content1,
                                   partnerB,tok2,lem2,content2,
                                   vocablist, 
                                   w2v_model_goog, w2v_model_twit, 
                                   bert_model,
                                   gpt_model
                                   ):

    # create empty dictionaries
    partner_direction = {}
    condition_info = {}
    semantic_W2V_goog = {}
    semantic_W2V_twit = {}
    utterance1_W2V = {}
    utterance2_W2V = {}
    # semantic_GPT = {}
    semantic_BERT = {}
    # utterance_length1 = {}
    # utterance_length2 = {}

    dictionaries_list = []

    # calculate conceptual alignment: word2vec: Google
    semantic_W2V_goog['semantic_W2V_goog'] = conceptual_alignment_w2v(lem1,lem2,vocablist,w2v_model_goog)
    dictionaries_list.append(semantic_W2V_goog.copy())

    # calculate conceptual alignment: word2vec: Twit  
    semantic_W2V_twit['semantic_W2V_twit'] = conceptual_alignment_w2v(lem1,lem2,vocablist,w2v_model_twit)
    dictionaries_list.append(semantic_W2V_twit.copy())

    # return utterances being compared: word2vec (just going to use Google for now)
    utterance1_W2V['utterance1_W2V'] = process_utterance_string(lem1,vocablist,w2v_model_goog)
    dictionaries_list.append(utterance1_W2V.copy())
    utterance2_W2V['utterance2_W2V'] = process_utterance_string(lem2,vocablist,w2v_model_goog)    
    dictionaries_list.append(utterance2_W2V.copy())   

    # calculate conceptual alignment: GPT    
    # semantic_GPT['semantic_GPT'] = conceptual_alignment_GPT(content1,content2,gpt_model)
    # dictionaries_list.append(semantic_GPT.copy())
    
    # # calculate conceptual alignment: BERT    
    semantic_BERT['semantic_BERT'] = conceptual_alignment_BERT(content1,content2,bert_model)
    dictionaries_list.append(semantic_BERT.copy())

    # determine directionality of leading/following comparison;  Note: Partner B is the lagged partner, thus, B is following A
    partner_direction['partner_direction'] = str(partnerA) + ">" + str(partnerB)
    dictionaries_list.append(partner_direction.copy())

    # add number of tokens in each utterance 
    # NOTE: For semantic, this is incorrect as a lot of tokens are removed in the vocablist filtering step, can just do this in R
    # utterance_length1['utterance_length1'] = len(lem1)
    # dictionaries_list.append(utterance_length1.copy())

    # utterance_length2['utterance_length2'] = len(lem2)
    # dictionaries_list.append(utterance_length2.copy())

    # add condition information
    condition_info['condition_info'] = cond_info
    dictionaries_list.append(condition_info.copy())

    # return alignment scores
    return dictionaries_list

def turn_by_turn_analysis(
                    dataframe,
                    delay,
                    vocablist,
                    w2v_model_goog,
                    w2v_model_twit,
                    bert_model,
                    gpt_model
                    ):

    # prepare the data to the appropriate type
    dataframe['token'] = dataframe['token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['lemma'] = dataframe['lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))

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
        content1=first_row['content'] ## NOTE: to be used with chatGPT and BERT; but could be cleaned for conjunctions and misspelled words?

        # grab all of lagged participant's data
        lagged_row = df_lagged.iloc[i]
        lagged_partner = lagged_row['participant']
        tok2=lagged_row['token']
        lem2=lagged_row['lemma']
        content2=lagged_row['content'] ## NOTE: to be used with chatGPT and BERT; but could be cleaned for conjunctions and misspelled words?

        # process multilevel alignment
        dictionaries_list=return_multilevel_alignment(cond_info=cond_info,
                                                         partnerA=first_partner,
                                                         tok1=tok1,lem1=lem1,
                                                         content1=content1,
                                                         partnerB=lagged_partner,
                                                         tok2=tok2,lem2=lem2,
                                                         content2=content2,
                                                         vocablist=vocablist,
                                                         w2v_model_goog=w2v_model_goog,
                                                         w2v_model_twit=w2v_model_twit,
                                                         bert_model=bert_model,
                                                         gpt_model=gpt_model)

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
                        delay,
                        concat_transcripts,
                        high_sd_cutoff,
                        low_n_cutoff,
                        model_input_file_w2v_twit,
                        model_input_file_w2v_goog,
                        model_id_bert,
                        model_id_gpt
                        ):

    # get the various semantic models to be used throughout
    # chatGPT
    
    # BERT
    bert_model = st.SentenceTransformer(model_id_bert) 
    
    # w2v_Google
    w2v_model_goog = load_w2v_trained(
                            pretrained_input_file=model_input_file_w2v_goog
                            )        
    
    # w2v_Twitter
    w2v_model_twit = load_w2v_trained(
                            pretrained_input_file=model_input_file_w2v_twit        )      
    
    # for w2v, need the filtered vocabulary list to be used to identify content words
    [unfiltered, filtered] = build_filtered_vocab(
                                            concat_transcripts=concat_transcripts,
                                            output_file_directory=output_file_directory,
                                            high_sd_cutoff=high_sd_cutoff,
                                            low_n_cutoff=low_n_cutoff
                                            )
    
    # time to begin looping through individual conversations and generating values
                                            
    # grab the files in the list
    file_list = glob.glob(input_files+"/*.txt")

    # create containers for alignment values
    tempT2T = list()
    # tempC2C = list()

    # cycle through each prepared file
    for fileName in file_list:

        # process the file if it's got a valid conversation
        dataframe=pd.read_csv(fileName, sep='\t',encoding='utf-8')
        if len(dataframe) > 1:

            # let us know which filename we're processing
            print(("Processing: "+fileName))

            # calculate turn-by-turn alignment scores
            xT2T=turn_by_turn_analysis(dataframe=dataframe,
                                         delay=delay,
                                         vocablist=filtered,
                                         w2v_model_goog=w2v_model_goog,
                                         w2v_model_twit=w2v_model_twit,
                                         bert_model=bert_model,
                                         gpt_model=model_id_gpt)
            tempT2T.append(xT2T)

        # if it's invalid, let us know
        else:
            print(("Invalid file: "+fileName))

    # update final dataframes
    AlignmentT2T = pd.concat(tempT2T)
    real_final_turn_df = AlignmentT2T.reset_index(drop=True)

    # export the final files
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M") 
    outfile = output_file_directory + f"/semantic_output_"+date_string+".txt"
    # outfile = output_file_directory + f"/{model_id_w2v}_{model_id_gpt}_{model_id_bert}_"+date_string+".txt"    
    real_final_turn_df.to_csv(outfile,                      
                      encoding='utf-8', index=False, sep='\t')

    return real_final_turn_df



# where are all the indiv txt files stored, and where to put the output generated from this analysis
INPUT_FILES = "/home/nduran4/ondemand/data/sys/myjobs/projects/default/14/prepped_stan/"
OUTPUT_FILES = "/home/nduran4/ondemand/data/sys/myjobs/projects/default/14/outputs-semantic/"

# set standards for which words will be evaluated in word2vec analysis
TRANSCRIPTS_CONCAT_VOCAB_FILE = "/home/nduran4/ondemand/data/sys/myjobs/projects/default/14/align_concatenated_dataframe.txt"
HIGH_SD_CUTOFF = None
LOW_N_CUTOFF = 1

# set standards to be used for real and surrogate
MAXLAG = 1

# for loading in the gensim w2v models
MODEL_w2v_google = 'word2vec-google-news-300'
MODEL_w2v_twitter = 'glove-twitter-200'

# for loading in the BERT and GPT models
MODEL_BERT = 'all-mpnet-base-v2'

MODEL_GPT = 'text-embedding-ada-002'
openai.api_key = "INSERT HERE"

turn_real = calculate_alignment(
                        input_files=INPUT_FILES,
                        output_file_directory=OUTPUT_FILES,
                        delay=MAXLAG,
                        concat_transcripts=TRANSCRIPTS_CONCAT_VOCAB_FILE,
                        high_sd_cutoff=HIGH_SD_CUTOFF,
                        low_n_cutoff=LOW_N_CUTOFF,
                        model_input_file_w2v_twit=MODEL_w2v_twitter,
                        model_input_file_w2v_goog=MODEL_w2v_google,
                        model_id_bert=MODEL_BERT,
                        model_id_gpt=MODEL_GPT
                        )   

EOF