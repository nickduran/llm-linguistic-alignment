# ALIGN: A Package for Linguistic Alignment Analysis

## Overview

ALIGN is a Python package designed to extract quantitative, reproducible metrics of linguistic alignment between speakers in conversational data. Linguistic alignment refers to the tendency of speakers to adopt similar linguistic patterns during conversation. This package provides tools to measure alignment at multiple linguistic levels:

- **Semantic alignment**: Using embedding models (BERT or FastText)
- **Lexical alignment**: Based on repeated words and phrases
- **Syntactic alignment**: Based on part-of-speech patterns

The package is designed to be flexible and extensible, allowing researchers to analyze conversations with different methodologies and compare results across different conversation types. It also includes functionality to generate "surrogate" conversation pairs to establish baseline alignment levels that would occur by chance.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Required Python Packages

- pandas
- numpy
- scikit-learn
- transformers
- torch
- gensim
- nltk
- tqdm
- python-dotenv

You can install all dependencies with:

```bash
pip install pandas numpy scikit-learn transformers torch gensim nltk tqdm python-dotenv
```

### Installing ALIGN

1. Clone the repository:
```bash
git clone https://github.com/your-username/align.git
cd align
```

2. Install the package in development mode:
```bash
pip install -e .
```

Alternatively, you can use the code directly from the cloned repository without installation.

## Getting Started

### Data Format

ALIGN expects conversation data in tab-separated text files with the following required columns:
- `participant`: IDs for the speakers
- `content`: The text of each utterance

Optional columns that enhance analysis:
- `token`: Tokenized utterances (list format)
- `lemma`: Lemmatized tokens (list format)
- `tagged_token`: Part-of-speech tagged tokens (list of tuples)
- `tagged_lemma`: Part-of-speech tagged lemmas (list of tuples)
- `tagged_stan_token`: Stanford-format POS tagged tokens (optional)
- `tagged_stan_lemma`: Stanford-format POS tagged lemmas (optional)

Example of a properly formatted file:
```
participant	content	token	lemma	tagged_token	tagged_lemma
PA:	hi there	["hi", "there"]	["hi", "there"]	[("hi", "UH"), ("there", "RB")]	[("hi", "UH"), ("there", "RB")]
PB:	hello how are you	["hello", "how", "are", "you"]	["hello", "how", "be", "you"]	[("hello", "UH"), ("how", "WRB"), ("are", "VBP"), ("you", "PRP")]	[("hello", "UH"), ("how", "WRB"), ("be", "VB"), ("you", "PRP")]
```

### Basic Usage

Here's a minimal example to analyze semantic alignment using BERT:

```python
from my_package.alignment import LinguisticAlignment

# Initialize the analyzer
analyzer = LinguisticAlignment(alignment_type="bert")

# Analyze a folder of conversation files
results = analyzer.analyze_folder(
    folder_path="path/to/your/conversation/files",
    output_directory="path/to/save/results",
    lag=1  # Number of turns to lag (default: 1)
)
```

### Comprehensive Example

Here's a more comprehensive example showing multiple analysis types and baseline comparison:

```python
import os
from my_package.alignment import LinguisticAlignment

# Define paths
data_path = "path/to/conversation/files"
output_folder = "path/to/results"

# Initialize with multiple alignment types
analyzer = LinguisticAlignment(
    alignment_types=["fasttext", "bert", "lexsyn"],
    cache_dir=os.path.join(output_folder, "cache")
)

# Configure parameters for FastText
fasttext_params = {
    "high_sd_cutoff": 3,    # Filter out words with frequency > mean + 3*std
    "low_n_cutoff": 2,      # Filter out words occurring < 2 times
    "save_vocab": True      # Save vocabulary lists to output directory
}

# Configure parameters for Lexical/Syntactic analysis
lexsyn_params = {
    "max_ngram": 3,         # Maximum n-gram size
    "ignore_duplicates": True,
    "add_stanford_tags": True  # Include Stanford POS tags if available
}

# Common parameters for all analyzers
common_params = {
    "lag": 1  # Number of turns to lag
}

# Analyze real conversations
real_results = analyzer.analyze_folder(
    folder_path=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params
)

# Parameters for surrogate generation
surrogate_params = {
    "all_surrogates": False,  # Generate a subset rather than all possible pairs
    "keep_original_turn_order": True,
    "id_separator": "_",
    "condition_label": "ExpBlock",  # Part of filename identifying experimental condition
    "dyad_label": "ASU-"  # Part of filename identifying conversation pair
}

# Analyze baseline (chance) alignment with surrogates
baseline_results = analyzer.analyze_baseline(
    input_files=data_path,
    output_directory=output_folder,
    **common_params,
    **fasttext_params,
    **lexsyn_params,
    **surrogate_params
)
```

## Analysis Types

### 1. Semantic Alignment with BERT

Uses BERT embeddings to measure semantic similarity between utterances:

```python
analyzer = LinguisticAlignment(alignment_type="bert")
```

Parameters:
- `model_name`: BERT model to use (default: "bert-base-uncased")
- `token`: Hugging Face token (optional)

### 2. Semantic Alignment with FastText

Uses FastText word embeddings to measure semantic similarity:

```python
analyzer = LinguisticAlignment(alignment_type="fasttext")
```

Parameters:
- `model_name`: FastText model (default: "fasttext-wiki-news-300")
- `high_sd_cutoff`: Standard deviation threshold for filtering high-frequency words
- `low_n_cutoff`: Minimum frequency threshold for filtering rare words
- `save_vocab`: Whether to save vocabulary lists to output directory

### 3. Lexical and Syntactic Alignment

Analyzes lexical and syntactic alignment using n-grams:

```python
analyzer = LinguisticAlignment(alignment_type="lexsyn")
```

Parameters:
- `max_ngram`: Maximum n-gram size (default: 2)
- `ignore_duplicates`: Whether to ignore duplicate n-grams (default: True)
- `add_stanford_tags`: Whether to include Stanford POS tags (default: False)

### 4. Multiple Analysis Types

Combine multiple analysis types in one run:

```python
analyzer = LinguisticAlignment(alignment_types=["bert", "fasttext", "lexsyn"])
# Or use all available analyzers
analyzer = LinguisticAlignment(alignment_types="all")
```

## Lag Parameter

The `lag` parameter determines how many turns to skip when pairing utterances. For example:

- `lag=1` (default): Each utterance is paired with the immediately following utterance
- `lag=2`: Each utterance is paired with the utterance that appears two positions later
- `lag=3`: Each utterance is paired with the utterance three positions later

This allows analysis of alignment patterns at different conversational distances.

## Surrogate Analysis

The package includes functionality to generate "surrogate" conversation pairs to establish baseline levels of alignment that would occur by chance:

```python
baseline_results = analyzer.analyze_baseline(
    input_files="path/to/conversation/files",
    output_directory="path/to/results",
    all_surrogates=False,  # Generate a subset rather than all possible pairs
    keep_original_turn_order=True
)
```

You can also reuse existing surrogate files:

```python
baseline_results = analyzer.analyze_baseline(
    input_files="path/to/conversation/files",
    output_directory="path/to/results",
    use_existing_surrogates="path/to/existing/surrogate/files"
)
```

## Output Files

ALIGN generates CSV files with detailed alignment metrics:

1. **Semantic alignment** (BERT):
   - `semantic_alignment_bert-base-uncased_lag1.csv`

2. **Semantic alignment** (FastText):
   - `semantic_alignment_fasttext-wiki-news-300_lag1_sd3_n2.csv`

3. **Lexical/syntactic alignment**:
   - `lexsyn_alignment_ngram2_lag1_noDups_noStan.csv`

4. **Merged results** (when using multiple analyzers):
   - `merged_alignment_results_lag1.csv`

5. **Baseline files** (surrogate analysis):
   - `baseline_alignment_bert-base-uncased_lag1.csv`
   - `baseline_alignment_fasttext_lag1_sd3_n2.csv`
   - `baseline_alignment_lexsyn_ngram2_lag1_noDups_noStan.csv`

## Hugging Face Token (Optional)

For BERT analysis, you can provide a Hugging Face token for accessing models. The token can be provided in several ways:

1. Directly in the code:
```python
analyzer = LinguisticAlignment(
    alignment_type="bert",
    token="your_huggingface_token"
)
```

2. As an environment variable `HUGGINGFACE_TOKEN`

3. In a config file at `~/.config/my_package/config.json`:
```json
{
    "huggingface_token": "your_huggingface_token"
}
```

## Detailed API Reference

### LinguisticAlignment

Main interface class for alignment analysis.

```python
LinguisticAlignment(
    alignment_type=None,   # Single alignment type
    alignment_types=None,  # Multiple alignment types
    **kwargs               # Additional configuration
)
```

#### Methods:

- **analyze_folder**(folder_path, output_directory=None, file_pattern="*.txt", lag=1, **kwargs)
  
  Analyze all text files in a folder.

- **analyze_baseline**(input_files, output_directory="results", surrogate_directory=None, use_existing_surrogates=None, **kwargs)
  
  Generate surrogate conversation pairs and analyze their alignment as a baseline.

- **process_file**(file_path, lag=1, **kwargs)
  
  Process a single file to compute alignment metrics.

### Common Parameters

These parameters can be used across different analyzer types:

- **lag**: Number of turns to lag when pairing utterances (default: 1)
- **file_pattern**: Pattern to match text files (default: "*.txt")
- **output_directory**: Directory to save results (optional)

## Caching

ALIGN uses caching to speed up repeated analyses:

1. **Model caching**: BERT and FastText models are cached in the specified `cache_dir`
2. **Embedding caching**: Computed embeddings are cached to avoid redundant computation

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   - Ensure all required packages are installed
   - For BERT analysis, ensure you have PyTorch installed

2. **File format issues**:
   - Verify your conversation files are tab-separated
   - Check that required columns are present and formatted correctly

3. **Memory issues**:
   - BERT models can use substantial memory; use a smaller model if needed
   - Process large datasets in smaller batches

### Getting Help

If you encounter issues not covered here, please submit an issue on GitHub.

## License

[Insert your license information here]

## Citation

If you use ALIGN in your research, please cite:

[Insert citation information here]