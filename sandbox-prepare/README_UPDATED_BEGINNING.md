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
│   └── preprocessing_example.py  # NEW: Phase 1 example
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

### Step 4: Download spaCy Language Model

**Required for transcript preprocessing with spaCy** (recommended for 100x speedup over Stanford):

```bash
python -m spacy download en_core_web_sm
```

**Note**: If you skip this step, you can still use NLTK-only preprocessing, or the package will auto-download the model the first time you use spaCy tagging. However, it's better to download it explicitly to avoid delays during your first preprocessing run.

### Step 5 (Optional): Set Up Stanford POS Tagger

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

**Note**: Most users won't need Stanford tagger. spaCy provides nearly identical accuracy (97.2% vs 97.4%) with 100x better speed.

### Step 6: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import align_test; print('✓ ALIGN package installed')"
python -c "import spacy; spacy.load('en_core_web_sm'); print('✓ spaCy model installed')"
```

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

```python
from align_test.alignment import LinguisticAlignment

# Initialize the analyzer
analyzer = LinguisticAlignment(alignment_type="bert")

# Analyze preprocessed conversation files
results = analyzer.analyze_folder(
    folder_path="./preprocessed",          # Your preprocessed files from Phase 1
    output_directory="./alignment_results", # Where to save results
    lag=1                                   # Number of turns to lag (default: 1)
)

print(f"Analyzed {len(results)} utterance pairs!")
```

#### Comprehensive Analysis Example

Here's a complete example showing the full pipeline from raw transcripts to alignment metrics:

```python
import os
from align_test.prepare_transcripts import prepare_transcripts
from align_test.alignment import LinguisticAlignment

# Define paths
raw_data = "./raw_transcripts"
preprocessed = "./preprocessed"
results_dir = "./alignment_results"

# Create directories
os.makedirs(preprocessed, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ============================================================
# PHASE 1: Preprocess transcripts
# ============================================================
print("Phase 1: Preprocessing transcripts...")

preprocessed_data = prepare_transcripts(
    input_files=raw_data,
    output_file_directory=preprocessed,
    run_spell_check=True,
    minwords=2,
    add_stanford_tags=True,
    stanford_tagger_type='spacy'  # Recommended: 100x faster than Stanford
)

print(f"Preprocessed {len(preprocessed_data)} utterances")

# ============================================================
# PHASE 2: Analyze alignment
# ============================================================
print("\nPhase 2: Analyzing linguistic alignment...")

# Initialize with multiple alignment types
analyzer = LinguisticAlignment(
    alignment_types=["fasttext", "bert", "lexsyn"],  # Analyze all three types
    cache_dir=os.path.join(results_dir, "cache")
)

# Configure parameters for each analyzer
fasttext_params = {
    "high_sd_cutoff": 3,    # Filter high-frequency words
    "low_n_cutoff": 2,      # Filter rare words
    "save_vocab": True      # Save vocabulary lists
}

lexsyn_params = {
    "max_ngram": 3,         # Maximum n-gram size
    "ignore_duplicates": True,
    "add_stanford_tags": True  # Use spaCy tags from preprocessing
}

common_params = {
    "lag": 1  # Analyze alignment at lag of 1 turn
}

# Analyze real conversations
real_results = analyzer.analyze_folder(
    folder_path=preprocessed,
    output_directory=results_dir,
    **common_params,
    **fasttext_params,
    **lexsyn_params
)

print(f"Analyzed {len(real_results)} utterance pairs")

# ============================================================
# PHASE 3 (Optional): Generate baseline with surrogates
# ============================================================
print("\nPhase 3: Generating baseline alignment with surrogates...")

# Parameters for surrogate generation
surrogate_params = {
    "all_surrogates": False,  # Generate subset of possible pairs
    "keep_original_turn_order": True,
    "id_separator": "-",            # Character separating filename parts
    "condition_label": "cond",      # Text identifying condition in filename
    "dyad_label": "time"            # Text identifying dyad in filename
}

# Analyze baseline (chance-level) alignment
baseline_results = analyzer.analyze_baseline(
    input_files=preprocessed,
    output_directory=results_dir,
    **common_params,
    **fasttext_params,
    **lexsyn_params,
    **surrogate_params
)

print(f"Baseline analysis complete!")
print(f"\nAll results saved to: {results_dir}")
```

#### Key Preprocessing Parameters

**File Processing**:
- `input_files`: Directory path or list of file paths
- `output_file_directory`: Where to save processed files
- `input_as_directory`: True if `input_files` is a directory path (default: True)
- `save_concatenated_dataframe`: Save combined file (default: True)

**Text Cleaning**:
- `minwords`: Minimum words per turn, removes shorter turns (default: 2)
- `use_filler_list`: List of filler words to remove (default: None = use regex)
- `filler_regex_and_list`: Use both regex and custom list (default: False)

**Spell-Checking**:
- `run_spell_check`: Enable Bayesian spell-checking (default: True)
- `training_dictionary`: Path to custom corpus (default: None = use Gutenberg)

**POS Tagging**:
- `add_stanford_tags`: Add `tagged_stan_*` columns (default: False)
- `stanford_tagger_type`: Which tagger to use - `'spacy'` or `'stanford'` (default: 'stanford')
- `spacy_model`: spaCy model name (default: 'en_core_web_sm')
- `stanford_pos_path`: Path to Stanford directory (required if using Stanford)
- `stanford_language_path`: Relative path to language model (required if using Stanford)
- `stanford_batch_size`: Batch size for Stanford (default: 50)

**Important**: The `minwords` parameter should be equal to or greater than the `max_ngram` you plan to use in alignment analysis. For example, if you'll use `max_ngram=3` in alignment analysis, set `minwords=3` during preprocessing.

#### POS Tagger Comparison

Choose the right tagger for your needs:

**Option 1: NLTK only** (No `add_stanford_tags` or set to `False`)
```python
prepare_transcripts(input_files="./raw", output_file_directory="./prep")
```
- ⚡ Fastest
- ✓ Good for lexical alignment only
- ✗ Limited syntactic alignment analysis

**Option 2: NLTK + spaCy** ⭐ **RECOMMENDED**
```python
prepare_transcripts(
    input_files="./raw", 
    output_file_directory="./prep",
    add_stanford_tags=True,
    stanford_tagger_type='spacy'
)
```
- ⚡ Very fast (~30% slower than NLTK only)
- ✓ Excellent accuracy (97.2%)
- ✓ Full syntactic alignment analysis
- ✓ Easy setup

**Option 3: NLTK + Stanford**
```python
prepare_transcripts(
    input_files="./raw",
    output_file_directory="./prep",
    add_stanford_tags=True,
    stanford_tagger_type='stanford',
    stanford_pos_path="/path/to/stanford-postagger-full-2020-11-17/",
    stanford_language_path="models/english-left3words-distsim.tagger"
)
```
- 🐌 Slow (~100x slower than spaCy)
- ✓ Highest accuracy (97.4%)
- ✓ Full syntactic alignment analysis
- ✗ Complex setup (requires Java + manual download)

**For 99% of users**: Use **Option 2 (spaCy)** for the best balance of speed and accuracy.

#### Transitioning from Raw Data to Alignment Results

Here's the complete workflow:

```python
# 1. Start with raw transcripts
#    Files: ./raw_transcripts/conv1.txt, conv2.txt, ...
#    Format: participant \t content

# 2. Preprocess (Phase 1)
from align_test.prepare_transcripts import prepare_transcripts

preprocessed = prepare_transcripts(
    input_files="./raw_transcripts",
    output_file_directory="./preprocessed",
    add_stanford_tags=True,
    stanford_tagger_type='spacy'  # Recommended
)
#    Output: ./preprocessed/conv1.txt, conv2.txt, ...
#    Format: participant \t content \t token \t lemma \t tagged_* \t file

# 3. Analyze alignment (Phase 2)
from align_test.alignment import LinguisticAlignment

analyzer = LinguisticAlignment(alignment_type="lexsyn")
results = analyzer.analyze_folder(
    folder_path="./preprocessed",        # Use preprocessed files
    output_directory="./results",
    lag=1,
    max_ngram=2,
    add_stanford_tags=True  # Use spaCy tags from preprocessing
)
#    Output: ./results/lexsyn_alignment_*.csv

# 4. Done! Results ready for statistical analysis
print(f"Analysis complete! {len(results)} utterance pairs analyzed")
```

---

## Analysis Types

(Continue with existing content from here...)
