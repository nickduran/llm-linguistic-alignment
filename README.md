# ALIGN: A Package for Linguistic Alignment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ALIGN is a Python package designed to extract quantitative, reproducible metrics of linguistic alignment between speakers in conversational data. Linguistic alignment refers to the tendency of speakers to adopt similar linguistic patterns during conversation. This package provides tools to measure alignment at multiple linguistic levels:

- **Semantic alignment**: Using embedding models (BERT or FastText)
- **Lexical alignment**: Based on repeated words and phrases
- **Syntactic alignment**: Based on part-of-speech patterns

The package is designed to be flexible and extensible, allowing researchers to analyze conversations with different methodologies and compare results across different conversation types. It also includes functionality to generate "surrogate" conversation pairs to establish baseline alignment levels that would occur by chance.

### Repository Structure

The ALIGN package has the following structure:

```
llm-linguistic-alignment/
├── src/                      # Source code
│   └── align_test/           # Core package files
│       ├── __init__.py
│       ├── alignment.py
│       ├── alignment_bert.py
│       ├── alignment_fasttext.py
│       ├── alignment_lexsyn.py
│       ├── bert_model.py
│       ├── fasttext_model.py
│       ├── config.py
│       ├── surrogates.py
│       └── data/             # Sample data
│           ├── prepped_stan_small/
│           │   └── [sample conversation files]
│           └── prepped_stan_mid/
│               └── [sample conversation files]
├── examples/                 # Example usage scripts
│   ├── basic_usage.py
│   └── advanced_usage.py
├── README.md
├── setup.py
├── requirements.txt
├── MANIFEST.in
├── LICENSE
└── .gitignore
```

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
git clone https://github.com/your-username/llm-linguistic-alignment.git
cd llm-linguistic-alignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

Alternatively, you can use the code directly from the cloned repository without installation by adding the `src` directory to your Python path.

## Getting Started

### Data Format

ALIGN expects conversation data in tab-separated text files with the following required columns:
- `participant`: IDs for the speakers
- `content`: The text of each utterance
- `token`: Tokenized utterances (list format)
- `lemma`: Lemmatized tokens (list format)
- `tagged_token`: Part-of-speech tagged tokens (list of tuples)
- `tagged_lemma`: Part-of-speech tagged lemmas (list of tuples)

Optional columns that enhance analysis:
- `tagged_stan_token`: Stanford-format POS tagged tokens (optional)
- `tagged_stan_lemma`: Stanford-format POS tagged lemmas (optional)

Example of a properly formatted file:
```
participant	content	token	lemma	tagged_token	tagged_lemma
PA:	hi there	["hi", "there"]	["hi", "there"]	[("hi", "UH"), ("there", "RB")]	[("hi", "UH"), ("there", "RB")]
PB:	hello how are you	["hello", "how", "are", "you"]	["hello", "how", "be", "you"]	[("hello", "UH"), ("how", "WRB"), ("are", "VBP"), ("you", "PRP")]	[("hello", "UH"), ("how", "WRB"), ("be", "VB"), ("you", "PRP")]
```

Sample conversation files are provided in the `src/align_test/data/prepped_stan_small` directory as examples of properly formatted input data. To use your own data, you can:

1. Create a new directory for your data
2. Specify the full path to your data directory when running the analysis

### Basic Usage

Here's a minimal example to analyze semantic alignment using BERT:

```python
import sys
import os

# Add src directory to path (if not installed)
sys.path.append("path/to/src")

from align_test.alignment import LinguisticAlignment

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
import sys

# Add src directory to path (if not installed)
sys.path.append("path/to/src")

from align_test.alignment import LinguisticAlignment

# Define paths
data_path = "path/to/conversation/files"
output_folder = "path/to/results"

# Initialize with multiple alignment types
analyzer = LinguisticAlignment(
    alignment_types=["fasttext", "bert", "lexsyn"],  # Run one or multiple analyzers
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
    "keep_original_turn_order": True,  # Maintain the sequential order of turns
    "id_separator": "_",  # Character separating parts of filename
    "condition_label": "ExpBlock",  # Text prefix identifying experimental conditions
    "dyad_label": "ASU-"  # Text prefix identifying participant/dyad IDs
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

**Note:** To use BERT analysis, you'll need to set up a Hugging Face token. See the [Setting Up Hugging Face Token](#setting-up-hugging-face-token) section at the end of this README for detailed instructions.

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

**Note:** When you run FastText analysis for the first time, the package will automatically download the FastText model and embeddings from the official repository. This download (approximately 1-2 GB) may take several minutes depending on your internet connection. The files are cached for future use.

### 3. Lexical and Syntactic Alignment

Analyzes lexical and syntactic alignment using n-grams:

```python
analyzer = LinguisticAlignment(alignment_type="lexsyn")
```

Parameters:
- `max_ngram`: Maximum n-gram size (default: 2)
- `ignore_duplicates`: Whether to ignore duplicate lexical n-grams when computing syntactic alignment (to address "lexical boost" effect)
- `add_stanford_tags`: Whether to include Stanford POS tags (default: False)

### 4. Multiple Analysis Types

Combine multiple analysis types in one run:

```python
analyzer = LinguisticAlignment(alignment_types=["bert", "fasttext", "lexsyn"])
```

## Lag Parameter

The `lag` parameter determines how many turns to skip when pairing utterances. For example:

- `lag=1` (default): Each utterance is paired with the immediately following utterance
- `lag=2`: Each utterance is paired with the utterance that appears two positions later
- `lag=3`: Each utterance is paired with the utterance three positions later

This allows analysis of alignment patterns at different conversational distances.

## Surrogate Analysis

### How Surrogate Generation Works

The surrogate generation process:
1. Takes participants from two different conversations
2. Pairs them together to create a new "artificial" conversation
3. Calculates alignment metrics for these artificial pairs
4. Uses these measurements as a baseline for chance-level alignment

For this to work properly, the surrogate generator needs to understand your filename structure to identify:
- Which participants belong to which conversation
- Which condition each conversation belongs to (as surrogates are only created within the same experimental condition)

### Configuring Surrogate Parameters for Your Dataset

The surrogate generation requires specific parameters that must match your filename structure. These parameters tell the algorithm how to extract participant/dyad IDs and condition information from your filenames.

```python
surrogate_params = {
    "id_separator": "_",  # Character separating parts of filename
    "dyad_label": "ASU-",  # Text prefix identifying participant/dyad IDs
    "condition_label": "ExpBlock",  # Text prefix identifying experimental conditions
    "all_surrogates": False,  # Whether to generate all possible pairs or a subset
    "keep_original_turn_order": True  # Whether to maintain original sequential turn order
}
```

#### Example Filename Formats and Corresponding Parameters

#### Example 1: Original Format
Filenames with structure: `ASU-T104_ExpBlock2-TrunkSlide.txt`

Correct parameters:

```python
surrogate_params = {
    "id_separator": "_",
    "dyad_label": "ASU-",
    "condition_label": "ExpBlock",
    "all_surrogates": False,
    "keep_original_turn_order": True
}
```
#### Example 2: Simple Time-Condition Format
Filenames with structure: `time191-cond1.txt`, `time192-cond1.txt`, etc.

Correct parameters:

```python
surrogate_params = {
    "id_separator": "-",
    "dyad_label": "time",
    "condition_label": "cond",
    "all_surrogates": False,
    "keep_original_turn_order": True
}
```
### Additional Parameters Explained

- **all_surrogates**:
  - `True`: Generate every possible combination of surrogate pairs (may be very large for big datasets)
  - `False`: Generate a smaller, representative subset of surrogate pairs

- **keep_original_turn_order**:
  - `True`: Maintain the sequential order of turns in the surrogate conversations (recommended)
  - `False`: Randomly shuffle the turns of each participant

### Using Existing Surrogate Files

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

## Setting Up Hugging Face Token

To use BERT for semantic alignment analysis, you'll need to set up a Hugging Face token. Here's how to do it:

### Step 1: Create a Hugging Face Account
1. Go to [Hugging Face](https://huggingface.co/) and sign up for an account if you don't have one
2. Log in to your account

### Step 2: Create an Access Token
1. Go to your [Hugging Face profile settings](https://huggingface.co/settings/tokens)
2. Click on "New token"
3. Give your token a name (e.g., "ALIGN_ACCESS")
4. Select "read" access
5. Click "Generate token"
6. Copy the generated token

### Step 3: Make the Token Available to ALIGN

Choose one of these methods to provide your token:

#### Option A: Environment Variable (Recommended)
Set an environment variable named `HUGGINGFACE_TOKEN`:

```bash
# On Linux/Mac
export HUGGINGFACE_TOKEN="your_token_here"

# On Windows (Command Prompt)
set HUGGINGFACE_TOKEN=your_token_here

# On Windows (PowerShell)
$env:HUGGINGFACE_TOKEN="your_token_here"
```

#### Option B: Config File
Create a config file at `~/.config/my_package/config.json`:

```bash
# Create directory if it doesn't exist
mkdir -p ~/.config/my_package
```

Then create the file with this content:
```json
{
    "huggingface_token": "your_token_here"
}
```

#### Option C: Provide in Code
Pass the token directly when initializing the analyzer:

```python
analyzer = LinguisticAlignment(
    alignment_type="bert",
    token="your_token_here"
)
```

### Step 4: Test Your Setup

Run the basic example to verify your token is working:

```python
import sys
import os

# Add src directory to path (if not installed)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "src"))

from align_test.alignment import LinguisticAlignment

# Initialize with token (if not provided via environment variable or config file)
analyzer = LinguisticAlignment(
    alignment_type="bert",
    # token="your_token_here"  # Uncomment if using Option C
)

# Use included sample data
data_path = os.path.join(project_root, "src", "align_test", "data", "prepped_stan_small")
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Run a test analysis
try:
    print("Testing BERT analyzer...")
    results = analyzer.analyze_folder(
        folder_path=data_path,
        output_directory=output_folder,
        lag=1
    )
    print("Success! BERT analyzer is working.")
except Exception as e:
    print(f"Error: {str(e)}")
    print("Please check your Hugging Face token and try again.")
```

If you encounter the error "401 Client Error: Unauthorized for url...", it means your token is invalid or not properly configured.

## License

MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.