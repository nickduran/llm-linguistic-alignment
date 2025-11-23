# ALIGN: A Package for Linguistic Alignment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ALIGN is a Python package designed to extract quantitative, reproducible metrics of linguistic alignment between speakers in conversational data. Linguistic alignment refers to the tendency of speakers to adopt similar linguistic patterns during conversation. This package provides a complete pipeline from raw transcripts to detailed alignment metrics:

**Phase 1: Transcript Preprocessing**
- Text cleaning and normalization
- Tokenization and lemmatization
- Part-of-speech tagging with multiple options (NLTK, spaCy, Stanford)
- Spell-checking (optional)

**Phase 2: Alignment Analysis**
- **Semantic alignment**: Using embedding models (BERT or FastText)
- **Lexical alignment**: Based on repeated words and phrases
- **Syntactic alignment**: Based on part-of-speech patterns
- **Baseline generation**: Create surrogate conversation pairs to establish chance-level alignment

The package is designed to be flexible and extensible, allowing researchers to analyze conversations with different methodologies and compare results across different conversation types.

### Repository Structure

The ALIGN package has the following structure:

```
llm-linguistic-alignment/
├── src/                      # Source code
│   └── align_test/           # Core package files
│       ├── __init__.py
│       ├── prepare_transcripts.py  # Phase 1: Preprocessing
│       ├── alignment.py            # Phase 2: Alignment analysis
│       ├── alignment_bert.py       
│       ├── alignment_fasttext.py
│       ├── alignment_lexsyn.py
│       ├── bert_model.py
│       ├── fasttext_model.py
│       ├── config.py
│       ├── surrogates.py
│       └── data/             # Sample data
│           ├── gutenberg.txt        # Spell-check corpus
│           ├── prepped_stan_small/  # Preprocessed samples
│           │   └── [sample conversation files]
│           └── prepped_stan_mid/    # Preprocessed samples
│               └── [sample conversation files]
├── examples/                 # Example usage scripts
│   ├── basic_usage.py
│   ├── advanced_usage.py
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
- Java 8+ (only required if using Stanford POS tagger)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/llm-linguistic-alignment.git
cd llm-linguistic-alignment
```

### Step 2: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs the required packages:
- pandas
- numpy
- scikit-learn
- transformers
- torch
- gensim
- nltk
- tqdm
- python-dotenv
- spacy

### Step 3: Install the ALIGN Package

```bash
pip install -e .
```

This installs the package in development/editable mode, meaning any changes you make to the source code will be immediately reflected without reinstalling.
<!-- 
### Step 4 (Optional): Set Up Stanford POS Tagger

**Only required if you want to use Stanford POS tagger** (slowest but highest accuracy option):

1. Download the Stanford POS Tagger from: https://nlp.stanford.edu/software/tagger.shtml#Download
   - Recommended version: `stanford-postagger-full-2020-11-17.zip`

2. Extract the downloaded file to a location on your computer:
   ```bash
   # Example on Linux/Mac:
   unzip stanford-postagger-full-2020-11-17.zip -d ~/tools/
   
   # Example on Windows:
   # Extract to C:\tools\stanford-postagger-full-2020-11-17\
   ```

3. Note the full path to the extracted directory. You'll need to provide two paths when using Stanford:
   - **Base directory path**: `/path/to/stanford-postagger-full-2020-11-17/`
   - **Language model path**: `models/english-left3words-distsim.tagger` (relative to base)

4. Verify Java is installed:
   ```bash
   java -version
   # Should show Java 8 or higher
   ```

**Note**: Most users won't need Stanford tagger. spaCy provides nearly identical accuracy (97.2% vs 97.4%) with 100x better speed. -->


## Getting Started

ALIGN provides a two-phase workflow for analyzing linguistic alignment in conversations:

1. **Phase 1: Preprocessing** - Convert raw transcripts to analysis-ready format
2. **Phase 2: Alignment Analysis** - Calculate alignment metrics and baselines

### Phase 1: Preprocessing Raw Transcripts

The first step is to preprocess your raw conversation transcripts. This prepares them for alignment analysis by cleaning text, tokenizing, lemmatizing, and adding part-of-speech tags.

#### Input Format for Raw Transcripts

Your raw transcript files should be **tab-separated text files** with two columns:

```
participant	content
Speaker1	Hello, how are you doing today?
Speaker2	I'm doing great, thanks for asking!
Speaker1	That's wonderful to hear.
Speaker2	How about you?
```

**Requirements**:
- **Tab-delimited** (not comma-separated)
- **UTF-8 encoding**
- **Column headers**: `participant` and `content`
- **One turn per row**
- **Chronological order** (rows should represent the temporal sequence of conversation)

**Filename Convention**:
Each conversation file should be named with identifiable components for dyad and condition:
- Example: `time200-cond1.txt`, `dyad5-condition2.txt`, `ASU-T104_ExpBlock2.txt`
- The filename format should be consistent across all files
- This matters later for surrogate generation (Phase 2)

#### Basic Preprocessing Example

Here's the simplest way to preprocess your transcripts:

```python
from align_test.prepare_transcripts import prepare_transcripts

# Preprocess all .txt files in a directory
results = prepare_transcripts(
    input_files="./raw_transcripts",      # Directory with your raw transcript files
    output_file_directory="./preprocessed" # Where to save processed files
)

print(f"Preprocessed {len(results)} utterances!")
```

**What this does**:
1. Cleans text (removes non-alphabetic characters, fillers like "um", "uh")
2. Merges adjacent turns by the same speaker
3. Performs spell-checking using Bayesian algorithm
4. Tokenizes text (expands contractions, splits into words)
5. Lemmatizes tokens (converts words to base forms)
6. Applies NLTK POS tagging
7. Saves processed files ready for alignment analysis

**Output**: Processed files with columns: `participant`, `content`, `token`, `lemma`, `tagged_token`, `tagged_lemma`, `file`

#### Recommended Preprocessing with spaCy

For better POS tagging accuracy with minimal speed impact, use spaCy:

```python
from align_test.prepare_transcripts import prepare_transcripts

results = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    add_stanford_tags=True,           # Add advanced POS tagging
    stanford_tagger_type='spacy'      # Use spaCy (100x faster than Stanford)
)
```

**What this adds**:
- Additional POS tags using spaCy's neural tagger
- Columns: `tagged_stan_token` and `tagged_stan_lemma` (for syntactic alignment analysis)
- Processing time: Only ~30% slower than NLTK-only mode
- Accuracy: 97.2% (vs 96.5% for NLTK, 97.4% for Stanford)

**When to use**: Recommended for most users who want syntactic alignment analysis

#### Advanced Preprocessing Options

**Disable spell-checking** (faster processing):
```python
results = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    run_spell_check=False  # Skip spell-checking
)
```

**Custom filler word removal**:
```python
results = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    use_filler_list=["um", "uh", "like", "you know"],  # Custom filler words
    filler_regex_and_list=True  # Use both regex and custom list
)
```

**Adjust minimum turn length**:
```python
results = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    minwords=3  # Require at least 3 words per turn (default: 2)
)
```

**Custom spell-check dictionary**:
```python
results = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    training_dictionary="./my_domain_corpus.txt"  # Use domain-specific dictionary
)
```

#### Using Stanford POS Tagger (Advanced)

**Only recommended if you need maximum accuracy and can tolerate slow processing** (~2 hours for 100 conversations vs ~1 minute with spaCy).

**Prerequisites**:
1. Download Stanford POS Tagger: https://nlp.stanford.edu/software/tagger.shtml#Download
2. Extract to a directory (e.g., `/home/user/tools/stanford-postagger-full-2020-11-17/`)
3. Verify Java is installed: `java -version`

**Understanding the Stanford Paths**:

The Stanford tagger requires two path parameters:

1. **`stanford_pos_path`**: The **full path to the Stanford tagger directory** 
   - This is the folder you extracted from the download
   - Should contain `stanford-postagger.jar` and a `models/` subdirectory
   - Example: `/home/user/tools/stanford-postagger-full-2020-11-17/`
   - **Important**: Include the trailing slash `/`

2. **`stanford_language_path`**: The **relative path to the language model file**
   - This path is relative to `stanford_pos_path`
   - For English, use: `models/english-left3words-distsim.tagger`
   - The full path will be: `stanford_pos_path` + `stanford_language_path`
   - Example full path: `/home/user/tools/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger`

**Visual Guide**:
```
stanford-postagger-full-2020-11-17/     ← stanford_pos_path points here
├── stanford-postagger.jar              ← Must be here
├── models/                             ← Contains language models
│   ├── english-left3words-distsim.tagger  ← stanford_language_path points here (relative)
│   ├── english-bidirectional-distsim.tagger
│   └── ... (other language models)
├── LICENSE.txt
└── README.txt
```

**Example Usage**:

```python
results = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    add_stanford_tags=True,
    stanford_tagger_type='stanford',
    
    # Path to Stanford directory (include trailing slash for clarity)
    stanford_pos_path="/home/user/tools/stanford-postagger-full-2020-11-17/",
    
    # Relative path to language model (from stanford_pos_path)
    stanford_language_path="models/english-left3words-distsim.tagger",
    
    # Optional: Adjust batch size (larger = faster but more memory)
    stanford_batch_size=50  # Process 50 utterances at a time
)
```

**Platform-Specific Examples**:

```python
# Linux/Mac:
stanford_pos_path="/home/username/tools/stanford-postagger-full-2020-11-17/"

# Windows:
stanford_pos_path="C:/tools/stanford-postagger-full-2020-11-17/"
# or
stanford_pos_path="C:\\tools\\stanford-postagger-full-2020-11-17\\"
```

**Performance Tips for Stanford**:
- Use `stanford_batch_size=100` if you have 8GB+ RAM (faster)
- Use `stanford_batch_size=25` if you have limited RAM (~4GB)
- Default of 50 works well for most systems

**Troubleshooting Stanford Setup**:

If you get an error like `FileNotFoundError: Stanford model not found`, verify:

```python
import os

# Check your paths
stanford_pos_path = "/home/user/tools/stanford-postagger-full-2020-11-17/"
stanford_language_path = "models/english-left3words-distsim.tagger"

# These should both be True:
model_full_path = os.path.join(stanford_pos_path, stanford_language_path)
jar_full_path = os.path.join(stanford_pos_path, "stanford-postagger.jar")

print(f"Model exists: {os.path.exists(model_full_path)}")
print(f"JAR exists: {os.path.exists(jar_full_path)}")

# If False, adjust your paths
```

#### Choosing a POS Tagger: Speed vs Accuracy

| Tagger | Processing Time* | Accuracy | Best For | Setup Required |
|--------|-----------------|----------|----------|----------------|
| **NLTK only** | ~30 seconds | 96.5% | Quick analysis, lexical alignment only | None |
| **NLTK + spaCy** ⭐ | ~1 minute | 97.2% | **Most users** (great speed/accuracy balance) | `spacy download` |
| **NLTK + Stanford** | ~1.5-2 hours | 97.4% | Maximum accuracy needed | Manual download + Java |

*For 100 conversations with ~50 utterances each (5,000 total utterances)

**Recommendation**: Use **spaCy** unless you have a specific need for the absolute highest accuracy. The 0.2% accuracy difference is negligible for most research purposes, while the 100x speedup is substantial.

#### Complete Preprocessing Example

```python
from align_test.prepare_transcripts import prepare_transcripts

# Full preprocessing with all options demonstrated
results = prepare_transcripts(
    # Input/Output
    input_files="./raw_transcripts",           # Directory containing raw .txt files
    output_file_directory="./preprocessed",     # Where to save processed files
    
    # Text Cleaning
    minwords=2,                                 # Minimum words per turn (default: 2)
    use_filler_list=None,                       # Use default regex filler removal
    filler_regex_and_list=False,                # Don't combine regex + custom list
    
    # Spell-Checking
    run_spell_check=True,                       # Enable spell-checking (default: True)
    training_dictionary=None,                   # Use default Gutenberg corpus
    
    # POS Tagging (RECOMMENDED: Use spaCy)
    add_stanford_tags=True,                     # Add advanced POS tags
    stanford_tagger_type='spacy',               # Use spaCy (fast and accurate)
    spacy_model='en_core_web_sm',              # spaCy model to use
    
    # File Handling
    input_as_directory=True,                    # Read all .txt files from directory
    save_concatenated_dataframe=True            # Save combined output file
)

print(f"Successfully preprocessed {len(results)} utterances!")
print(f"Output files saved to: ./preprocessed")
print(f"Ready for alignment analysis!")
```

#### Output Files from Preprocessing

After running `prepare_transcripts()`, you'll have:

1. **Individual processed files** (one per input file):
   - `preprocessed/time200-cond1.txt`
   - `preprocessed/time210-cond1.txt`
   - Each contains all columns needed for alignment analysis

2. **Concatenated file** (optional, if `save_concatenated_dataframe=True`):
   - `preprocessed/align_concatenated_dataframe.txt`
   - Combines all conversations into a single file

**Example processed file structure**:
```
participant	content	token	lemma	tagged_token	tagged_lemma	tagged_stan_token	tagged_stan_lemma	file
cgv	okay	"['okay']"	"['okay']"	"[('okay', 'UH')]"	"[('okay', 'UH')]"	"[('okay', 'UH')]"	"[('okay', 'UH')]"	time200-cond1.txt
kid	im sitting over here	"['im', 'sitting', 'over', 'here']"	"['im', 'sit', 'over', 'here']"	"[('im', 'VBP'), ('sitting', 'VBG'), ('over', 'IN'), ('here', 'RB')]"	...
```

**Note**: All list and tuple columns are stored as string representations (e.g., `"['word1', 'word2']"`) that can be parsed with Python's `ast.literal_eval()`. This format is required for compatibility with the alignment analysis phase.

---

### Phase 2: Analyzing Alignment in Preprocessed Data

Once you've preprocessed your transcripts (Phase 1), you can analyze linguistic alignment using the preprocessed files.

#### Data Format Expected by Alignment Analysis

The alignment analysis phase expects preprocessed files with these columns:
- `participant`: IDs for the speakers
- `content`: The cleaned text of each utterance
- `token`: Tokenized utterances (list format stored as string)
- `lemma`: Lemmatized tokens (list format stored as string)
- `tagged_token`: Part-of-speech tagged tokens (list of tuples stored as string)
- `tagged_lemma`: Part-of-speech tagged lemmas (list of tuples stored as string)

Optional columns that enhance syntactic alignment analysis:
- `tagged_stan_token`: Advanced POS tagged tokens (from spaCy or Stanford)
- `tagged_stan_lemma`: Advanced POS tagged lemmas (from spaCy or Stanford)

**Note**: If you preprocessed your data using Phase 1 above, your files are already in the correct format! If you're using pre-existing data, ensure it matches this format.

Sample preprocessed files are provided in the `src/align_test/data/prepped_stan_small` directory as examples.

#### Basic Alignment Analysis

Here's a minimal example to analyze semantic alignment using BERT:

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