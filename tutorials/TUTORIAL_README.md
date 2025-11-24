# ALIGN Package Tutorials

This directory contains user-friendly tutorial notebooks for the ALIGN (Analyzing Linguistic Alignment) package.

## Tutorial Notebooks

### 📘 Tutorial 1: Preprocessing (`tutorial_1_preprocessing.ipynb`)
**Purpose**: Learn how to prepare raw conversational transcripts for alignment analysis

**What's Included:**
- Step-by-step preprocessing workflow
- Using different POS taggers (NLTK, spaCy, Stanford)
- Setup instructions for optional taggers
- Input/output format validation
- Sample data inspection

**Time to Complete**: ~10-15 minutes (plus download time for optional taggers)

---

### 📗 Tutorial 2: Alignment Analysis (`tutorial_2_alignment.ipynb`)
**Purpose**: Learn how to analyze linguistic alignment in preprocessed conversations

**What's Included:**
- Lexical-syntactic alignment (word and grammar similarity)
- Semantic alignment with FastText
- Semantic alignment with BERT (optional)
- Multi-analyzer comprehensive analysis
- Visualization and interpretation
- Correlation analysis between alignment types

**Time to Complete**: ~15-20 minutes (plus download time for FastText on first run)

---

### 📙 Tutorial 3: Baseline Analysis (`tutorial_3_baseline.ipynb`)
**Purpose**: Learn how to establish baseline alignment levels using surrogate conversations

**What's Included:**
- Understanding surrogate/baseline analysis
- Generating surrogate conversation pairs
- Analyzing alignment in surrogate data
- Comparing real vs. baseline alignment
- Statistical significance testing
- Interpreting results

**Why This Matters:**
- Establishes what alignment occurs "by chance"
- Allows statistical testing of real alignment
- Essential for research and publication
- Helps interpret whether observed alignment is meaningful

**Time to Complete**: ~20-30 minutes (generates many surrogate pairs)

---

## Quick Start

### Step 1: Clone and Install
```bash
git clone https://github.com/your-username/align2-linguistic-alignment.git
cd align2-linguistic-alignment
pip install -r requirements.txt
pip install -e .
```

### Step 2: Open Tutorial 1
```bash
jupyter notebook tutorial_1_preprocessing.ipynb
```

> **💡 Tip**: You can also open and run these notebooks in **Visual Studio Code**! VS Code has excellent Jupyter notebook support with features like IntelliSense, debugging, and variable inspection. Just open the `.ipynb` file in VS Code and click "Run All" or run cells individually.

### Step 3: Follow Along
- View the notebook on GitHub to see expected outputs
- Download and run locally to process your own data
- Use included CHILDES sample data to learn

### Step 4: Open Tutorial 2
```bash
jupyter notebook tutorial_2_alignment.ipynb
```

### Step 5: Analyze!
- Use preprocessed data from Tutorial 1
- Compute alignment metrics
- Visualize and interpret results

### Step 6: Open Tutorial 3 (Optional but Recommended)
```bash
jupyter notebook tutorial_3_baseline.ipynb
```

### Step 7: Compare Real vs. Baseline!
- Generate surrogate conversation pairs
- Compute baseline alignment levels
- Test if real alignment is statistically significant
- Publish with confidence!

---

## What's Included in Each Tutorial

### Tutorial 1 Output:
```
tutorial_output/
├── preprocessed_nltk/          # NLTK-only (fastest)
├── preprocessed_spacy/         # NLTK + spaCy (recommended)
└── preprocessed_stanford/      # NLTK + Stanford (highest accuracy)
```

### Tutorial 2 Output:
```
tutorial_output/alignment_results/
├── lexsyn/                     # Lexical-syntactic alignment results
│   ├── lexsyn_alignment_ngram2_lag1_noDups_noAdd.csv
│   └── lexsyn_alignment_ngram2_lag1_noDups_withSpacy.csv
├── fasttext/                   # FastText semantic alignment
│   └── semantic_alignment_fasttext_lag1_sd3_n1.csv
├── bert/                       # BERT semantic alignment (optional)
│   └── semantic_alignment_bert-base-uncased_lag1.csv
├── merged/                     # Combined multi-analyzer results
│   └── merged-lag1-ngram2-noAdd-noDups-sd3-n1.csv
└── cache/                      # Model caches (FastText, BERT)
```

### Tutorial 3 Output:
```
tutorial_output/baseline_results/
├── surrogates/                 # Generated surrogate conversation pairs
│   └── surrogate_run-{timestamp}/
│       ├── SurrogatePair-dyad1A-dyad2B-cond1.txt
│       ├── SurrogatePair-dyad1A-dyad3B-cond1.txt
│       └── ... (many surrogate pairs)
├── lexsyn/                     # Baseline alignment results
│   └── baseline_alignment_lexsyn_ngram2_lag1_noDups_noAdd.csv
├── fasttext/                   # Baseline semantic alignment
│   └── baseline_alignment_fasttext_lag1_sd3_n1.csv
└── comparison/                 # Real vs. Baseline comparisons
    └── alignment_comparison_lexsyn.csv
```

---

## Using Your Own Data

### Input Format Requirements:
- Tab-delimited text files (`.txt`)
- Required columns: `participant`, `content`
- UTF-8 encoding
- One utterance per row

### Example Input:
```
participant	content
Speaker1	Hello there
Speaker2	Hi how are you
Speaker1	I am doing well
```

### To Use Your Data:
1. **Tutorial 1**: Change `INPUT_DIR` to your data directory
2. Run preprocessing
3. **Tutorial 2**: Update `INPUT_DIR_NLTK` to your preprocessed output
4. Run alignment analysis
