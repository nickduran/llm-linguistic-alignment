# ALIGN 2.0: A Modern Package for Linguistic Alignment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** This is ALIGN 2.0, a complete revamp of the original ALIGN package. For information about the original version (still available on PyPI), see [original Github repository](https://github.com/nickduran/align-linguistic-alignment) and the [2019 Psychological Methods paper](https://dynamicog.org/publications/journal/25-Duran2019.pdf) by Duran, Paxton, & Fusaroli.

## Overview

ALIGN 2.0 is a comprehensive Python package for measuring **linguistic alignment** in conversational data—the tendency of speakers to adopt similar linguistic patterns during interaction. This package provides flexible tools to quantify alignment at multiple linguistic levels:

- **Semantic alignment**: Using modern embedding models (BERT or FastText)
- **Lexical alignment**: Based on repeated words and phrases (n-grams)
- **Syntactic alignment**: Based on part-of-speech patterns

The package now includes enhanced preprocessing capabilities, support for multiple POS taggers, IMPROVED surrogate generation for baseline comparisons, and increased flexibility for diverse research needs.

---

## 🚀 Quick Start

**New to ALIGN?** The fastest way to get started is through our comprehensive Jupyter notebook tutorials:

1. **[Tutorial 1: Preprocessing](https://github.com/nickduran/align2-linguistic-alignment/blob/main/tutorials/tutorial_1_preprocessing.ipynb)**  
   Learn how to transform raw conversational transcripts into analysis-ready format

2. **[Tutorial 2: Alignment Analysis](https://github.com/nickduran/align2-linguistic-alignment/blob/main/tutorials/tutorial_2_alignment.ipynb)**  
   Discover how to measure linguistic alignment at multiple levels

3. **[Tutorial 3: Baseline/Surrogate Analysis](https://github.com/nickduran/align2-linguistic-alignment/blob/main/tutorials/tutorial_3_baseline.ipynb)**  
   Generate surrogate conversation pairs to establish chance-level baseline alignment for statistical comparison with real conversations

These tutorials provide hands-on, step-by-step guidance using real conversational data and are the recommended starting point for all users.

---

## 📥 Installation

**Important:** ALIGN 2.0 is not yet on PyPI. The [original ALIGN](https://github.com/nickduran/align-linguistic-alignment) remains available there, but to use ALIGN 2.0, you must install from GitHub.

### Prerequisites
- Python 3.7+ (tested with Python 3.13)
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/nickduran/align2-linguistic-alignment.git
cd align2-linguistic-alignment
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install in Editable Mode

This allows you to modify the code and see changes immediately:

```bash
pip install -e .
```

---

## 🆕 What's New in ALIGN 2.0

ALIGN 2.0 represents a complete modernization while maintaining compatibility with the original methodology ([Duran, Paxton, & Fusaroli, 2019](https://dynamicog.org/publications/journal/25-Duran2019.pdf). Here's what's different:


### Major Enhancements

✅ **Modern language models**: Native integration of BERT for contextualized semantic embeddings—no manual model downloads required

✅ **Dramatically faster processing**: spaCy POS tagging is 100-200x faster than Stanford tagger, with minimal accuracy trade-offs

✅ **Streamlined setup**: No external dependencies to manually download (Stanford tagger optional)—BERT and FastText models download automatically via Hugging Face and Gensim

✅ **Enhanced surrogate generation**: More flexible baseline creation with better control over pairing logic and turn order preservation

✅ **Multiple POS tagger support**: Choose between NLTK, spaCy, or Stanford taggers, or compare them side-by-side in the same analysis

✅ **Robust preprocessing pipeline**: Comprehensive data validation, spell-checking, and cleaning with detailed error messages when issues arise

✅ **Progress visualization**: Real-time progress bars (via tqdm) for all long-running operations—no more wondering if it's working

✅ **Production-ready code**: Full type hints, comprehensive docstrings, and modular architecture for easier extension and debugging

✅ **Interactive tutorials**: Step-by-step Jupyter notebooks with real conversational data to get you started quickly

### Core Methodology Preserved

The fundamental alignment calculations remain consistent with the original ALIGN:
- Cosine similarity for semantic and lexical alignment
- N-gram based lexical and syntactic analysis  
- Surrogate generation for baseline comparisons
- Turn-by-turn directionality tracking
- Support for multiple lag values

---

## 📚 Detailed Documentation

### Two-Phase Workflow

ALIGN 2.0 uses a clear two-phase approach:

#### **Phase 1: Preprocessing** (`prepare_transcripts.py`)

Transforms raw conversational data into analysis-ready format:

```python
from align_test.prepare_transcripts import prepare_transcripts

# Basic usage: NLTK only (fastest, default)
prepare_transcripts(
    input_files="path/to/raw/transcripts",           # Directory containing .txt files
    output_file_directory="path/to/preprocessed",    # Where to save processed files
    run_spell_check=True,                            # Enable spell-checking
    minwords=2,                                      # Minimum words per turn
    use_filler_list=None                             # Use default filler removal
)

# With spaCy tagging (recommended for speed)
prepare_transcripts(
    input_files="path/to/raw/transcripts",
    output_file_directory="path/to/preprocessed",
    run_spell_check=True,
    minwords=2,
    add_additional_tags=True,                        # Enable additional tagger
    tagger_type='spacy',                             # Use spaCy for additional tags
    spacy_model='en_core_web_sm'                     # spaCy model to use
)

# With Stanford tagging (slowest, most accurate)
prepare_transcripts(
    input_files="path/to/raw/transcripts",
    output_file_directory="path/to/preprocessed",
    run_spell_check=True,
    minwords=2,
    add_additional_tags=True,                        # Enable additional tagger
    tagger_type='stanford',                          # Use Stanford for additional tags
    stanford_pos_path="/path/to/stanford-postagger-full-2020-11-17/",
    stanford_language_path="models/english-left3words-distsim.tagger"
)
```

**Important Notes:**
- Input files **must** be tab-delimited with columns named `participant` and `content`
- NLTK tagging is always included (base columns: `tagged_token`, `tagged_lemma`)
- Additional tagger columns are added only if `add_additional_tags=True`:
  - spaCy: `tagged_spacy_token`, `tagged_spacy_lemma`
  - Stanford: `tagged_stanford_token`, `tagged_stanford_lemma`
- Text cleaning (non-ASCII removal, filler removal) happens automatically

**Key Parameters:**
- `input_files`: Directory path containing raw transcript files
- `output_file_directory`: Directory where preprocessed files will be saved
- `run_spell_check`: Enable/disable automatic spell-checking (default: True)
- `minwords`: Minimum number of words per turn (shorter turns are removed; default: 2)
- `add_additional_tags`: Add second set of POS tags beyond NLTK (default: False)
- `tagger_type`: Which additional tagger to use—`'spacy'` or `'stanford'` (default: 'stanford')
- `use_filler_list`: Custom list of fillers to remove (None = use default regex)

**Output Files:**
Each conversation produces a processed file with these columns:
- `participant`: Speaker IDs
- `content`: Cleaned utterance text
- `token`: Tokenized words (string representation of list)
- `lemma`: Lemmatized tokens (string representation of list)
- `tagged_token`: NLTK POS-tagged tokens (string representation of list of tuples)
- `tagged_lemma`: NLTK POS-tagged lemmas (string representation of list of tuples)
- `tagged_spacy_token`: spaCy POS-tagged tokens (if `add_additional_tags=True` and `tagger_type='spacy'`)
- `tagged_spacy_lemma`: spaCy POS-tagged lemmas (if `add_additional_tags=True` and `tagger_type='spacy'`)
- `tagged_stanford_token`: Stanford POS-tagged tokens (if `add_additional_tags=True` and `tagger_type='stanford'`)
- `tagged_stanford_lemma`: Stanford POS-tagged lemmas (if `add_additional_tags=True` and `tagger_type='stanford'`)

---

#### **Phase 2: Alignment Analysis** (`alignment.py`)

Calculates alignment metrics on preprocessed data:

```python
from align_test.alignment import LinguisticAlignment

# Initialize analyzer
analyzer = LinguisticAlignment(
    alignment_types=["bert", "fasttext", "lexsyn"]
)

# Analyze conversations
results = analyzer.analyze_folder(
    folder_path="path/to/preprocessed/files",
    output_directory="path/to/results",
    lag=1,
    # FastText parameters
    high_sd_cutoff=3,
    low_n_cutoff=2,
    # Lexical/syntactic parameters
    max_ngram=3,
    ignore_duplicates=True,
    add_additional_tags=True             # Use additional POS tags if available
)
```

**Key Parameters:**
- `lag`: Number of turns between paired utterances (default: 1)
- `max_ngram`: Maximum n-gram size for lexical/syntactic analysis (default: 2)
- `ignore_duplicates`: Remove lexical overlap from syntactic patterns (default: True)
- `add_additional_tags`: Whether to use additional POS tags from preprocessing (default: False)
- `additional_tagger_type`: Which additional tagger columns to use—`'spacy'` or `'stanford'`
- `high_sd_cutoff`: FastText high-frequency word filter (default: 3)
- `low_n_cutoff`: FastText low-frequency word filter (default: 1)

**Note:** The `add_additional_tags` and `additional_tagger_type` parameters tell the analyzer which POS tag columns to use from your preprocessed files. These must match what you created during preprocessing.

---

### Analysis Types

#### 1. Semantic Alignment with BERT

Measures semantic similarity using contextualized embeddings:

```python
analyzer = LinguisticAlignment(alignment_type="bert")
results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="results/",
    model_name="bert-base-uncased",  # or other BERT variants
    lag=1
)
```

**Setup Required:** See [Hugging Face Token Setup](#-setting-up-hugging-face-token) section below.

**Output:** `semantic_alignment_bert-base-uncased_lag1.csv`

---

#### 2. Semantic Alignment with FastText

Measures semantic similarity using static word embeddings:

```python
analyzer = LinguisticAlignment(alignment_type="fasttext")
results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="results/",
    model_name="fasttext-wiki-news-300",
    high_sd_cutoff=3,    # Filter high-frequency words
    low_n_cutoff=2,      # Filter rare words
    save_vocab=True,     # Save vocabulary lists
    lag=1
)
```

**Output:** `semantic_alignment_fasttext-wiki-news-300_lag1_sd3_n2.csv`

---

#### 3. Lexical & Syntactic Alignment

Measures word and phrase repetition (lexical) and grammatical structure reuse (syntactic):

```python
analyzer = LinguisticAlignment(alignment_type="lexsyn")

# Using NLTK tags only
results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="results/",
    max_ngram=3,                      # Analyze uni-, bi-, and trigrams
    ignore_duplicates=True,           # Remove lexical overlap from syntax
    add_additional_tags=False,        # Use only NLTK tags
    lag=1
)

# Using additional tags (spaCy)
results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="results/",
    max_ngram=3,
    ignore_duplicates=True,
    add_additional_tags=True,         # Use additional tags
    additional_tagger_type='spacy',   # Specify which additional tagger
    lag=1
)

# Using additional tags (Stanford)
results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="results/",
    max_ngram=3,
    ignore_duplicates=True,
    add_additional_tags=True,
    additional_tagger_type='stanford',
    lag=1
)
```

**Output Filenames:**
- With NLTK only: `lexsyn_alignment_ngram3_lag1_noDups_noAdd.csv`
- With spaCy: `lexsyn_alignment_ngram3_lag1_noDups_withSpacy.csv`
- With Stanford: `lexsyn_alignment_ngram3_lag1_noDups_withStan.csv`

**Key Parameters:**
- `max_ngram`: Maximum n-gram size (2=bigrams, 3=trigrams, etc.)
- `ignore_duplicates`: If `True`, removes syntactic patterns that share lexical content
- `add_additional_tags`: Whether to use additional POS tags (default: False)
- `additional_tagger_type`: Which additional tagger to use: `'spacy'` or `'stanford'`

---

#### 4. Multiple Analysis Types

Run all analyses together:

```python
analyzer = LinguisticAlignment(
    alignment_types=["bert", "fasttext", "lexsyn"]
)

results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="results/",
    lag=1,
    # FastText-specific
    high_sd_cutoff=3,
    low_n_cutoff=2,
    # Lexsyn-specific
    max_ngram=3,
    ignore_duplicates=True,
    add_additional_tags=True,
    additional_tagger_type='spacy'
)
```

**Output:** Individual CSV files for each analysis type, plus `merged_alignment_results_lag1.csv`

---

### Understanding the Lag Parameter

The `lag` parameter controls which utterances are paired for alignment calculation:

- `lag=1` (default): Each utterance paired with the immediately following one
- `lag=2`: Each utterance paired with the utterance 2 positions later
- `lag=3`: Each utterance paired with the utterance 3 positions later

Example with `lag=1`:
```
Turn 1: "I love pizza"              →  paired with → Turn 2: "Me too"
Turn 2: "Me too"                    →  paired with → Turn 3: "What's your favorite?"
Turn 3: "What's your favorite?"     →  paired with → Turn 4: "Pepperoni"
```

This allows analysis of alignment at different conversational distances.

---

## 🎯 Surrogate (Baseline) Analysis

Surrogate analysis creates artificial conversation pairs to establish chance-level baseline alignment. This is crucial for determining whether observed alignment exceeds what would occur randomly.

### How It Works

1. Takes participants from **different** real conversations
2. Pairs them to create artificial dyads
3. Calculates alignment metrics for these fabricated pairs
4. Provides baseline for statistical comparison

### Configuring for Your Dataset

The surrogate generator needs to parse your filenames to identify participants and experimental conditions. Configure these parameters to match your naming scheme:

```python
from align_test.alignment import LinguisticAlignment

analyzer = LinguisticAlignment(alignment_types=["bert", "lexsyn"])

# Analyze baseline alignment
baseline_results = analyzer.analyze_baseline(
    input_files="preprocessed_data/",
    output_directory="results/baseline/",
    # Surrogate configuration
    id_separator="_",                      # Character separating filename parts
    dyad_label="ASU-",                     # Prefix identifying participant/dyad IDs
    condition_label="ExpBlock",            # Prefix identifying experimental conditions
    all_surrogates=False,                  # Generate subset (True = all combinations)
    keep_original_turn_order=True,         # Maintain sequential turn order (recommended)
    # Analysis parameters (must match your real data analysis)
    lag=1,
    max_ngram=3,
    ignore_duplicates=True,
    add_additional_tags=True,
    additional_tagger_type='spacy'
)
```

### Filename Structure Examples

#### Example 1: Research Lab Format
Filenames: `ASU-T104_ExpBlock2-TrunkSlide.txt`

```python
surrogate_params = {
    "id_separator": "_",
    "dyad_label": "ASU-",
    "condition_label": "ExpBlock"
}
```

Parsing logic:
- `ASU-T104` → Dyad ID (T104)
- `ExpBlock2` → Condition (2)
- Surrogates only pair participants from same condition

#### Example 2: Simple Format
Filenames: `dyad23_condition1.txt`

```python
surrogate_params = {
    "id_separator": "_",
    "dyad_label": "dyad",
    "condition_label": "condition"
}
```

#### Example 3: Timestamp Format
Filenames: `time191-cond1.txt`

```python
surrogate_params = {
    "id_separator": "-",
    "dyad_label": "time",
    "condition_label": "cond"
}
```

### Understanding Surrogate Parameters

- **`all_surrogates`**:
  - `False`: Generate representative subset (faster, usually sufficient)
  - `True`: Generate every possible pairing (computationally expensive)

- **`keep_original_turn_order`**:
  - `True`: Preserve sequential turn order (recommended—maintains temporal structure)
  - `False`: Randomly shuffle turns (less conservative baseline)

### Reusing Existing Surrogates

If you've already generated surrogate files:

```python
baseline_results = analyzer.analyze_baseline(
    input_files="preprocessed_data/",
    output_directory="results/baseline/",
    use_existing_surrogates="path/to/surrogate/files/",
    lag=1,
    # Include other analysis parameters as needed
    max_ngram=3,
    add_additional_tags=True,
    additional_tagger_type='spacy'
)
```

### Output Files

Baseline analysis generates files with `baseline_` prefix:
- `baseline_alignment_bert-base-uncased_lag1.csv`
- `baseline_alignment_fasttext_lag1_sd3_n2.csv`
- `baseline_alignment_lexsyn_ngram3_lag1_noDups_withSpacy.csv`

---

## 📊 Understanding Output Files

### File Naming Convention

ALIGN 2.0 uses descriptive filenames that encode analysis parameters:

**Format:** `[prefix]_alignment_[model]_lag[N]_[params].csv`

Examples:
- `semantic_alignment_bert-base-uncased_lag1.csv`
- `semantic_alignment_fasttext-wiki-news-300_lag1_sd3_n2.csv`
- `lexsyn_alignment_ngram3_lag1_noDups_withSpacy.csv`
- `baseline_alignment_bert-base-uncased_lag1.csv`

### Output Columns

All alignment result files share these core columns:

| Column | Description |
|--------|-------------|
| `file` | Source conversation filename |
| `order` | Turn sequence number (0-indexed) |
| `direction` | Who follows whom (e.g., "PA→PB" or "PB→PA") |
| `participant_lead` | ID of leading speaker |
| `participant_follow` | ID of following speaker |
| `turn_lead` | Utterance from leading speaker |
| `turn_follow` | Utterance from following speaker |

**Alignment scores** (additional columns depend on analysis type):
- **BERT/FastText**: `semantic_similarity` (range: -1 to 1, typically 0.3-0.9)
- **Lexical**: `lexical_sim_ngram[N]` for each n-gram size (range: 0-1)
- **Syntactic**: `syntactic_sim_ngram[N]` for each n-gram size (range: 0-1)

### Merged Output (Multiple Analyzers)

When running multiple analysis types together, ALIGN generates:

`merged_alignment_results_lag[N].csv`

This file combines all alignment metrics in a single row per turn pair, making it easy to:
- Compare alignment across linguistic levels
- Perform multi-level statistical modeling
- Visualize relationships between alignment types

---

## 🔑 Setting Up Hugging Face Token

BERT-based semantic alignment requires a (free) Hugging Face account and access token.

### Step 1: Create Account
1. Go to [huggingface.co](https://huggingface.co/) and sign up
2. Log in to your account

### Step 2: Generate Token
1. Visit [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "ALIGN_ACCESS")
4. Select "read" permission
5. Generate and copy the token

### Step 3: Provide Token to ALIGN

Choose one method:

#### Option A: Environment Variable (Recommended)

```bash
# Linux/Mac
export HUGGINGFACE_TOKEN="your_token_here"

# Windows Command Prompt
set HUGGINGFACE_TOKEN=your_token_here

# Windows PowerShell
$env:HUGGINGFACE_TOKEN="your_token_here"
```

#### Option B: Configuration File

Create `~/.config/my_package/config.json`:

```json
{
    "huggingface_token": "your_token_here"
}
```

#### Option C: Pass Directly in Code

```python
analyzer = LinguisticAlignment(
    alignment_type="bert",
    token="your_token_here"
)
```

### Testing Your Setup

```python
from align_test.alignment import LinguisticAlignment

# Test BERT with minimal data
analyzer = LinguisticAlignment(alignment_type="bert")

# Use any preprocessed conversation files
results = analyzer.analyze_folder(
    folder_path="preprocessed_data/",
    output_directory="test_output/",
    lag=1
)

print("✓ BERT analyzer working correctly!")
```

If you see "401 Client Error: Unauthorized", double-check your token configuration.

---

## 💡 Usage Examples

### Example 1: Basic Semantic Alignment

```python
from align_test.alignment import LinguisticAlignment

# Simplest possible analysis
# Note: Requires HUGGINGFACE_TOKEN environment variable for BERT
analyzer = LinguisticAlignment(alignment_type="bert")

results = analyzer.analyze_folder(
    folder_path="preprocessed_conversations/",  # Already preprocessed files
    output_directory="results/",
    lag=1  # Pair each utterance with the next one
)

print(f"Analyzed {len(results)} conversation files")
# Output: results/semantic_alignment_bert-base-uncased_lag1.csv
```

---

### Example 2: Complete Multi-Level Analysis with Baselines

```python
from align_test.alignment import LinguisticAlignment

# Initialize with all analysis types
analyzer = LinguisticAlignment(
    alignment_types=["bert", "fasttext", "lexsyn"],
    cache_dir="cache/"  # Store downloaded models here
)

# Configure parameters (can reuse for both real and baseline)
common_params = {
    "lag": 1
}

fasttext_params = {
    "high_sd_cutoff": 3,    # Filter high-frequency words
    "low_n_cutoff": 2,      # Filter rare words
    "save_vocab": True      # Save vocabulary to output
}

lexsyn_params = {
    "max_ngram": 3,            # Analyze up to 3-word phrases
    "ignore_duplicates": True,  # Remove lexical overlap from syntax
    "add_additional_tags": True  # Use spaCy/Stanford tags (must exist in preprocessed files)
}

# Analyze real conversations
real_results = analyzer.analyze_folder(
    folder_path="preprocessed_conversations/",
    output_directory="results/real/",
    **common_params,
    **fasttext_params,
    **lexsyn_params
)

# Configure surrogate generation
surrogate_params = {
    "id_separator": "_",           # e.g., "dyad5_condition1.txt"
    "dyad_label": "dyad",          # Filename prefix for dyad ID
    "condition_label": "condition", # Filename prefix for condition
    "all_surrogates": False,       # Sample ~50% of possible pairs
    "keep_original_turn_order": True  # Maintain temporal structure
}

# Generate baseline with surrogates
baseline_results = analyzer.analyze_baseline(
    input_files="preprocessed_conversations/",
    output_directory="results/baseline/",
    **common_params,
    **fasttext_params,
    **lexsyn_params,
    **surrogate_params
)

print("Analysis complete!")
print(f"Real conversations: {len(real_results)} files")
print(f"Baseline surrogates: {len(baseline_results)} files")
```

---

### Example 3: Comparing Different Lag Values

```python
from align_test.alignment import LinguisticAlignment

analyzer = LinguisticAlignment(alignment_type="lexsyn")

# Analyze at different conversational distances
# Note: Each lag value requires a separate analysis run
for lag_value in [1, 2, 3]:
    print(f"\nAnalyzing with lag={lag_value}...")
    print(f"  - lag=1: consecutive turns")
    print(f"  - lag=2: skip 1 turn between pairs")
    print(f"  - lag=3: skip 2 turns between pairs")
    
    results = analyzer.analyze_folder(
        folder_path="preprocessed_conversations/",
        output_directory=f"results/lag{lag_value}/",
        lag=lag_value,
        max_ngram=2,
        ignore_duplicates=True,
        add_additional_tags=True  # Use additional POS tags if available
    )
    
    print(f"  ✓ Results saved to: results/lag{lag_value}/")
    print(f"    Files analyzed: {len(results)}")

print("\n✓ All lag analyses complete!")
print("Compare results across lag values to see how alignment changes with distance.")
```

---

### Example 4: Complete Preprocessing and Analysis Workflow

```python
from align_test.prepare_transcripts import prepare_transcripts
from align_test.alignment import LinguisticAlignment

# ============================================================
# PHASE 1: Preprocess raw transcripts
# ============================================================
print("PHASE 1: Preprocessing raw conversations...")
print("  Input: Tab-separated files with 'participant' and 'content' columns")
print("  Output: Preprocessed files with tokens, lemmas, and POS tags\n")

preprocessed_results = prepare_transcripts(
    input_files="raw_transcripts/",
    output_file_directory="preprocessed/",
    run_spell_check=True,
    minwords=2,
    add_additional_tags=True,  # Add spaCy POS tags
    tagger_type='spacy',       # Use spaCy (fast and accurate)
    spacy_model='en_core_web_sm'
)

print(f"✓ Preprocessed {len(preprocessed_results)} utterances")
print(f"  Saved to: preprocessed/\n")

# ============================================================
# PHASE 2: Analyze alignment
# ============================================================
print("PHASE 2: Analyzing linguistic alignment...")
print("  Analyzers: BERT (semantic) + Lexical/Syntactic\n")

analyzer = LinguisticAlignment(alignment_types=["bert", "lexsyn"])

results = analyzer.analyze_folder(
    folder_path="preprocessed/",
    output_directory="results/",
    lag=1,
    max_ngram=3,
    ignore_duplicates=True,
    add_additional_tags=True  # Use spaCy columns from preprocessing phase
)

print(f"\n✓ Analysis Complete! Analyzed {len(results)} files\n")
print("Output files:")
print("  - results/bert/semantic_alignment_bert-base-uncased_lag1.csv")
print("  - results/lexsyn/lexsyn_alignment_ngram3_lag1_noDups_withSpacy.csv")
print("  - results/merged_alignment_results_lag1.csv")
```

---

## 📁 Repository Structure

```
align2-linguistic-alignment/
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
│       └── data/               # Sample data
│           ├── gutenberg.txt   # Spell-check corpus
│           ├── CHILDES/        # 20 properly formatted input files for preprocessing
├── tutorials/                # Example usage scripts
│   ├── TUTORIAL_README.md
|   ├── tutorial_1_preprocessing.ipynb
|   ├── tutorial_2_alignment.ipynb
│   ├── tutorial_3_baseline.ipynb
├── README.md
├── setup.py
├── requirements.txt
├── MANIFEST.in
├── LICENSE
```

---

## 🔬 Methodological Notes

### Alignment Calculation

ALIGN 2.0 uses **cosine similarity** to measure alignment across all linguistic levels:

- **Semantic (BERT/FastText)**: Cosine similarity between utterance embeddings
- **Lexical**: Cosine similarity between n-gram frequency vectors
- **Syntactic**: Cosine similarity between POS n-gram frequency vectors

**Why cosine similarity?**
1. **Interpretable**: Values from -1 to 1 (or 0 to 1 for n-grams)
2. **Length-normalized**: Controls for utterance length differences
3. **Established**: Widely used in NLP and information retrieval
4. **Consistent**: Same metric across all linguistic levels

### Directionality

ALIGN tracks alignment directionality separately:
- **PA→PB**: How much does PB align with PA?
- **PB→PA**: How much does PA align with PB?

This allows research on:
- Leader-follower dynamics
- Power relationships
- Conversational roles

### Statistical Considerations

When comparing real vs. baseline alignment:

1. **Match parameters**: Use identical settings for real and surrogate analyses
2. **Aggregate appropriately**: Consider conversation-level or turn-level aggregation
3. **Account for non-independence**: Turns within conversations are related
4. **Use mixed-effects models**: Account for random effects (e.g., dyad, individual)

Example statistical approach:
```R
# In R with lme4
library(lme4)

model <- lmer(
    alignment ~ data_type * condition + (1|dyad) + (1|turn_order),
    data = combined_data
)
```

---

## 📖 Citation

If you use ALIGN 2.0 in your research, please cite the original methodology paper:

```bibtex
@article{duran2019align,
  title={ALIGN: Analyzing Linguistic Interactions with Generalizable techNiques—A Python Library},
  author={Duran, Nicholas D and Paxton, Alexandra and Fusaroli, Riccardo},
  journal={Psychological Methods},
  year={2019},
  publisher={American Psychological Association},
  doi={10.1037/met0000206}
}
```

**Note:** A paper describing ALIGN 2.0 enhancements is in preparation. Check this repository for updates.

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "No module named 'align_test'"**
- **Solution**: Ensure you installed in editable mode: `pip install -e .`

**Issue: "401 Client Error: Unauthorized" (BERT)**
- **Solution**: Check your Hugging Face token configuration (see [setup section](#-setting-up-hugging-face-token))

**Issue: "FileNotFoundError: [Errno 2] No such file or directory"**
- **Solution**: Verify file paths are correct and files exist

**Issue: "ValueError: No valid conversation files found"**
- **Solution**: Check that preprocessed files have required columns (`participant`, `content`, `token`, etc.)

**Issue: "KeyError: 'tagged_spacy_lemma'"**
- **Solution**: Ensure preprocessing included `add_additional_tags=True` or specify correct `tagger` parameter

**Issue: Slow BERT analysis**
- **Solution**: BERT is compute-intensive. Consider:
  - Using FastText for initial exploration
  - Running on GPU-enabled machine
  - Processing smaller batches

**Issue: Surrogate generation fails**
- **Solution**: Verify `id_separator`, `dyad_label`, and `condition_label` match your filename structure exactly

### Getting Help

1. Check the [tutorials](tutorials/) for complete working examples
2. Review this README carefully
3. Open an issue on [GitHub](https://github.com/your-username/llm-linguistic-alignment/issues)
4. Include error messages, code snippets, and Python version

---

## 🤝 Contributing

We welcome contributions! This package is under active development. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

**Areas for contribution:**
- Additional preprocessing options
- New alignment metrics
- Performance optimizations
- Documentation improvements
- Bug fixes

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Nicholas D. Duran

---

## 🙏 Acknowledgments

ALIGN 2.0 builds upon the original ALIGN methodology developed by Nicholas D. Duran, Alexandra Paxton, and Riccardo Fusaroli (2019).

The package leverages several excellent open-source projects:
- [Hugging Face Transformers](https://huggingface.co/transformers/) for BERT models
- [Gensim](https://radimrehurek.com/gensim/) for FastText embeddings
- [NLTK](https://www.nltk.org/) for NLP utilities
- [spaCy](https://spacy.io/) for linguistic processing
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) for POS tagging

---

## 📞 Contact

For questions about ALIGN 2.0:
- **GitHub Issues**: [Open an issue](https://github.com/your-username/llm-linguistic-alignment/issues)
- **Email**: nicholas.duran@utexas.edu

For questions about the original ALIGN methodology, please refer to the [2019 paper](https://dynamicog.org/publications/journal/25-Duran2019.pdf).

---

**Ready to start?** Head to the [Quick Start](#-quick-start) section or jump directly to the [Phase 1 tutorial](tutorials/test_comprehensive_prepare.ipynb)!
