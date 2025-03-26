import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_package.alignment import SemanticAlignment

# Define path to data folder - adjust paths as needed for your environment
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "my_package", "data", "prepped_stan_small")
output_folder = "tests/results"

# Make sure output directories exist
os.makedirs(output_folder, exist_ok=True)
# os.makedirs(os.path.join(output_folder, "bert"), exist_ok=True)
# os.makedirs(os.path.join(output_folder, "word2vec"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "lexsyn"), exist_ok=True)

# Initialize analyzers for all three methods
# bert_analyzer = SemanticAlignment(embedding_model="bert")
# w2v_analyzer = SemanticAlignment(
#     embedding_model="word2vec",
#     cache_dir=os.path.join(output_folder, "word2vec", "cache")
# )
lexsyn_analyzer = SemanticAlignment(embedding_model="lexsyn")

# # Run analyses with appropriate parameters
# print("Running BERT semantic alignment analysis...")
# bert_results = bert_analyzer.analyze_folder(
#     folder_path=data_path,
#     output_directory=os.path.join(output_folder, "bert"),
#     lag=1
# )

# print("Running Word2Vec semantic alignment analysis...")
# w2v_results = w2v_analyzer.analyze_folder(
#     folder_path=data_path,
#     output_directory=os.path.join(output_folder, "word2vec"),
#     lag=1,
#     high_sd_cutoff=3,  # Filter out words with frequency > mean + 3*std
#     low_n_cutoff=1,    # Filter out words occurring < 1 times
#     save_vocab=True    # Save vocabulary lists to output directory
# )

print("Running lexical and syntactic alignment analysis...")
lexsyn_results = lexsyn_analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=os.path.join(output_folder, "lexsyn"),
    lag=1,
    max_ngram=3,
    ignore_duplicates=True,
    add_stanford_tags=True  # Set to False if Stanford tags aren't available
)