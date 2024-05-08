#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 16            # number of cores 
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
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
import pickle

# Third-party modules
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import sentence_transformers as st

# Built-in modules from specific classes
from os import listdir
from os.path import isfile, join

# Function to get lagged conversational turns
def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df['utter1'] = df['content']
    df['utter2'] = df['content'].shift(-1)
    df['utter_order'] = df['participant'] + ' ' + df['participant'].shift(-1)
    return df

# Function to compute embeddings, but first checks if already in cache and if not, add them there afterward
default_embedding_engine = "text-embedding-ada-002"  # text-embedding-ada-002 is recommended
def get_embedding_with_cache(
    text: str,
    engine: str = default_embedding_engine
) -> list:
    # Skip if there is no text content for computing embedding
    if text is None:
        return None
    if (text, engine) not in gpt_embedding_cache.keys():
        # if not in cache, call API to get embedding
        gpt_embedding_cache[(text, engine)] = openai.embeddings.create(input=[text], model=engine).data[0].embedding
    return gpt_embedding_cache[(text, engine)]

# Function to process and get embeddings/cosines for a single file
def process_file(file_path, gpt_embedding_cache, default_embedding_engine):       
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    df = process_input_data(df)

   # Create column of embeddings
    for column in ["utter1", "utter2"]:
        df[f"{column}_embedding"] = df[column].apply(get_embedding_with_cache)
    
    # Create column of cosine similiarity
    df["cosine_similarity"] = df.apply(
        lambda row: cosine_similarity(
            np.array(row["utter1_embedding"]).reshape(1, -1),
            np.array(row["utter2_embedding"]).reshape(1, -1)
        )[0][0] if row["utter1_embedding"] is not None and row["utter2_embedding"] is not None else None,
        axis=1
    )

    return df

# where are all the indiv txt files stored, and where to put the output generated from this analysis
FOLDER_PATH = "/home/nduran4/ondemand/data/sys/myjobs/projects/default/11/prepped_stan/"
text_files = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f)) and f.endswith('.txt')]
OUTPUT_FILES = "/home/nduran4/ondemand/data/sys/myjobs/projects/default/11/outputs-semantic/"

# set standards to be used for real and surrogate
MAXLAG = 1

# for loading in the BERT and GPT models
MODEL_BERT = 'all-mpnet-base-v2'

MODEL_GPT = 'text-embedding-ada-002'
openai.api_key = "INSERT HERE"

# Load or initialize the GPT embedding cache, if none there (first time running), then create empty cache to build
gpt_embedding_cache_path = "data/gpt_embedding_cache.pkl"
try:
    with open(gpt_embedding_cache_path, "rb") as f:
        gpt_embedding_cache = pickle.load(f)
except FileNotFoundError:
    gpt_embedding_cache = {}


# Process each file and update the cache
concatenated_df = pd.DataFrame()
for file_name in text_files:
    file_path = os.path.join(FOLDER_PATH, file_name)
    df = process_file(file_path, gpt_embedding_cache, default_embedding_engine)
    concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

    # Save the updated embedding cache to disk after processing each file
    with open(gpt_embedding_cache_path, "wb") as gpt_embedding_cache_file:
        pickle.dump(gpt_embedding_cache, gpt_embedding_cache_file)




EOF