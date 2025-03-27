# ALIGN: Semantic Alignment Analyzer

ALIGN is a Python package that extracts quantitative, reproducible metrics of semantic alignment between speakers in naturalistic language corpora. The package analyzes conversations by measuring semantic similarity between utterances using embedding-based techniques.

## Installation

```bash
pip install align-nlp
```

## Basic Usage

```python
from align import SemanticAlignment

# Initialize analyzer with Word2Vec
analyzer = SemanticAlignment(embedding_model="word2vec")

# Process files with a lag of 1 turn
results = analyzer.analyze_folder(
    folder_path="path/to/conversation/files",
    output_directory="results",
    lag=1
)
```

## Cache Management

ALIGN uses large language models like Word2Vec that need to be downloaded once and cached for future use. By default, these models are stored in a cache directory in your home folder (`~/.cache/align/models/`).

### Setting a Custom Cache Directory

You can specify a custom location for model files directly when creating the analyzer:

```python
analyzer = SemanticAlignment(
    embedding_model="word2vec",
    cache_dir="/path/to/custom/cache"  # Specify your preferred location
)
```

This is particularly useful if:
- You have limited space in your home directory
- You want to share models across multiple users
- You prefer to keep all project files in one location

The cache directory will be created if it doesn't already exist.

## Supported Models

ALIGN currently supports two types of embeddings:

- **BERT**: High-quality contextual embeddings (`embedding_model="bert"`)
- **Word2Vec**: Fast, lightweight word embeddings (`embedding_model="word2vec"`)

## Word2Vec Configuration

When using Word2Vec, you can control vocabulary filtering:

```python
results = analyzer.analyze_folder(
    folder_path="path/to/conversation/files",
    output_directory="results",
    lag=1,
    high_sd_cutoff=3,     # Filter out words with frequency > mean + 3*std
    low_n_cutoff=2,       # Filter out words occurring < 2 times
    save_vocab=True       # Save vocabulary lists to output directory
)
```

## Output

ALIGN produces CSV files containing similarity scores between utterance pairs, along with detailed embedding information. If `save_vocab=True` with Word2Vec, you'll also get vocabulary frequency lists in the output directory.

## Example

```python
import os
from align import SemanticAlignment

# Initialize Word2Vec analyzer with default cache location
analyzer = SemanticAlignment(embedding_model="word2vec")

# Process all .txt files in a folder
results = analyzer.analyze_folder(
    folder_path="./conversation_data",
    output_directory="./alignment_results",
    lag=1  # Compare each utterance with the next one
)

# Show results
print(f"Processed {len(results)} utterance pairs")
print(f"Average similarity: {results['word2vec-google-news-300_cosine_similarity'].mean()}")
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- tqdm
- gensim (for Word2Vec)
- transformers (for BERT)

## Citation

If you use ALIGN in your research, please cite:
```
[Citation information here]
```