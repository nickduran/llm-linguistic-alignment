# prepare_transcripts.py Analysis Report
## Focus: Output Format Compatibility & Stanford Tagger Issues

---

## EXECUTIVE SUMMARY

The `prepare_transcripts.py` file has THREE CRITICAL ISSUES that prevent seamless integration with the alignment analysis phase:

1. ❌ **Data structures stored as Python objects instead of string representations**
2. ⚠️ **Stanford tagger path handling is fragile and error-prone**
3. ⚠️ **No progress indication or error handling for long-running operations**

All issues are fixable with targeted code modifications.

---

## DETAILED FINDINGS

### Issue 1: Data Structure Storage Format (CRITICAL)

#### Problem
The current code stores lists and tuples as **native Python objects** when saving to CSV:
```python
# Current code (lines 400-401)
df['tagged_token'] = df['token'].apply(nltk.pos_tag)
df['tagged_lemma'] = df['lemma'].apply(nltk.pos_tag)
```

This produces CSV cells containing actual Python objects:
```
[('hello', 'UH'), ('how', 'WRB'), ('are', 'VBP')]  # Python list object
```

#### Required Format
The alignment scripts expect **string representations** that can be parsed with `ast.literal_eval()`:
```
"[('hello', 'UH'), ('how', 'WRB'), ('are', 'VBP')]"  # String representation
```

#### Evidence
From your existing alignment scripts:
- `alignment_lexsyn.py`, line 19: uses `ast.literal_eval(x)` to parse
- `alignment_fasttext.py`, line 137: uses `ast.literal_eval(tokens)` to parse
- `alignment_bert.py`: expects string inputs

#### Impact
**CRITICAL**: Without this fix, the preprocessed files CANNOT be loaded by the alignment analysis scripts. You'll get errors like:
```
ValueError: malformed node or string
```

#### Solution
Convert all list/tuple columns to string representations BEFORE saving:

```python
# Store original objects temporarily
df['token_obj'] = df['token'].copy()
df['lemma_obj'] = df['lemma'].copy()

# Convert to string representations
df['token'] = df['token'].apply(str)
df['lemma'] = df['lemma'].apply(str)
df['tagged_token'] = df['token_obj'].apply(lambda x: str(nltk.pos_tag(x)))
df['tagged_lemma'] = df['lemma_obj'].apply(lambda x: str(nltk.pos_tag(x)))

# For Stanford tags
if add_stanford_tags:
    df['tagged_stan_token'] = df['token_obj'].apply(lambda x: str(stanford_tagger.tag(x)))
    df['tagged_stan_lemma'] = df['lemma_obj'].apply(lambda x: str(stanford_tagger.tag(x)))

# Drop temporary columns
df = df.drop(['token_obj', 'lemma_obj'], axis=1)
```

---

### Issue 2: Stanford Tagger Path Handling (HIGH PRIORITY)

#### Problem 1: Path Concatenation is Fragile
Current code (lines 396-397):
```python
stanford_tagger = StanfordPOSTagger(
    stanford_pos_path + stanford_language_path,
    stanford_pos_path + 'stanford-postagger.jar'
)
```

**Issues:**
- If `stanford_pos_path` doesn't end with `/`, paths will be malformed:
  - User provides: `/path/to/stanford`
  - Code creates: `/path/to/stanfordmodels/english...` ❌
  - Should be: `/path/to/stanford/models/english...` ✅

- Not cross-platform compatible (Windows uses `\`, Unix uses `/`)

#### Problem 2: No Path Validation
If files don't exist, you get cryptic Java errors instead of clear Python errors:
```
Exception: Stanford POS Tagger java class not found
```
Instead of:
```
FileNotFoundError: Stanford model not found at: /expected/path/model.tagger
```

#### Problem 3: Unclear Error Messages
Users need to know:
- Which exact paths were attempted
- Whether Java is installed
- Whether the Stanford tagger version is compatible

#### Solution
Use `os.path` for robust path handling with validation:

```python
import os

if add_stanford_tags:
    # Validate parameters
    if stanford_pos_path is None or stanford_language_path is None:
        raise ValueError(
            'To use Stanford POS tagger, specify both:\n'
            '  - stanford_pos_path: directory path\n'
            '  - stanford_language_path: relative model path\n'
            'Example:\n'
            '  stanford_pos_path="/path/to/stanford-postagger-full-2020-11-17/"\n'
            '  stanford_language_path="models/english-left3words-distsim.tagger"'
        )
    
    # Normalize and construct paths
    stanford_pos_path = os.path.normpath(stanford_pos_path)
    model_path = os.path.join(stanford_pos_path, stanford_language_path)
    jar_path = os.path.join(stanford_pos_path, 'stanford-postagger.jar')
    
    # Validate files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Stanford model not found at: {model_path}\n"
            f"Check:\n"
            f"  1. stanford_pos_path: {stanford_pos_path}\n"
            f"  2. stanford_language_path: {stanford_language_path}"
        )
    
    if not os.path.exists(jar_path):
        raise FileNotFoundError(
            f"Stanford JAR not found at: {jar_path}\n"
            f"Check stanford_pos_path: {stanford_pos_path}"
        )
    
    # Create tagger with error handling
    try:
        stanford_tagger = StanfordPOSTagger(model_path, jar_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Stanford tagger: {e}\n"
            f"Possible causes:\n"
            f"  1. Java not installed or not in PATH\n"
            f"  2. Incompatible Stanford tagger version\n"
            f"  3. Incorrect paths"
        )
```

---

### Issue 3: User Experience Issues

#### Problem 1: No Progress Indication
Stanford tagging is VERY slow (potentially hours for large datasets), but there's no progress bar or status updates. Users have no idea if it's working or frozen.

#### Solution
Add progress bars using `tqdm`:
```python
from tqdm import tqdm
tqdm.pandas(desc="Stanford POS tagging")

df['tagged_stan_token'] = df['token_obj'].progress_apply(
    lambda x: str(stanford_tagger.tag(x))
)
```

#### Problem 2: No Error Recovery
If Stanford tagging fails partway through, the entire process crashes. Should handle errors gracefully:

```python
try:
    df['tagged_stan_token'] = df['token_obj'].progress_apply(
        lambda x: str(stanford_tagger.tag(x))
    )
except Exception as e:
    print(f"WARNING: Stanford tagging failed: {e}")
    print("Continuing without Stanford tags...")
    df['tagged_stan_token'] = str([])
```

---

## VERIFICATION CHECKLIST

After implementing fixes, verify:

✅ **Output Format**
1. Load a preprocessed file with `pd.read_csv(file, sep='\t')`
2. Check that `df['token'].iloc[0]` is a STRING starting with `"["`
3. Verify `ast.literal_eval(df['token'].iloc[0])` produces a list
4. Confirm the same for `lemma`, `tagged_token`, `tagged_lemma` columns

✅ **Stanford Tagger** (if using)
1. Test with missing Stanford directory → should get clear error
2. Test with wrong model path → should get clear error
3. Test with correct paths → should see progress bar
4. Verify `tagged_stan_token` and `tagged_stan_lemma` columns exist
5. Verify they contain string representations of lists

✅ **Integration Test**
1. Run `prepare_transcripts` on sample data
2. Load output files
3. Run alignment analysis on preprocessed files
4. Should complete without errors

---

## ADDITIONAL RECOMMENDATIONS

### 1. Add Input Validation
```python
def validate_input_file(filepath):
    """Validate that input file has required columns"""
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    required_cols = ['participant', 'content']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file missing required columns: {missing}\n"
            f"File: {filepath}\n"
            f"Found columns: {df.columns.tolist()}"
        )
```

### 2. Add Column Ordering
```python
# At end of ApplyPOSTagging function
column_order = [
    'participant', 'content', 'token', 'lemma',
    'tagged_token', 'tagged_lemma'
]
if add_stanford_tags:
    column_order.extend(['tagged_stan_token', 'tagged_stan_lemma'])
column_order.append('file')

df = df[[col for col in column_order if col in df.columns]]
```

### 3. Add Example Usage Documentation
```python
"""
Example Usage
-------------

Basic usage (no Stanford tagger):
>>> from align_test import prepare_transcripts
>>> results = prepare_transcripts(
...     input_files="./raw_data",
...     output_file_directory="./preprocessed_data"
... )

With Stanford tagger:
>>> results = prepare_transcripts(
...     input_files="./raw_data",
...     output_file_directory="./preprocessed_data",
...     add_stanford_tags=True,
...     stanford_pos_path="/path/to/stanford-postagger-full-2020-11-17/",
...     stanford_language_path="models/english-left3words-distsim.tagger"
... )
"""
```

---

## TESTING RECOMMENDATIONS

### Test 1: Basic Functionality
```python
# Test with your sample files
results = prepare_transcripts(
    input_files="./",
    output_file_directory="./test_output",
    add_stanford_tags=False
)

# Verify output format
test_file = "./test_output/time200-cond1.txt"
df = pd.read_csv(test_file, sep='\t')
assert isinstance(df['token'].iloc[0], str)
assert isinstance(ast.literal_eval(df['token'].iloc[0]), list)
```

### Test 2: Stanford Tagger
```python
# Test error handling with bad paths
try:
    results = prepare_transcripts(
        input_files="./",
        output_file_directory="./test_output",
        add_stanford_tags=True,
        stanford_pos_path="/nonexistent/path/",
        stanford_language_path="models/english-left3words-distsim.tagger"
    )
    assert False, "Should have raised FileNotFoundError"
except FileNotFoundError as e:
    print(f"✓ Got expected error: {e}")
```

### Test 3: Integration with Alignment
```python
# Preprocess data
prepare_transcripts(
    input_files="./raw_data",
    output_file_directory="./preprocessed_data"
)

# Run alignment analysis
from align_test.alignment import LinguisticAlignment
analyzer = LinguisticAlignment(alignment_type="lexsyn")
results = analyzer.analyze_folder(
    folder_path="./preprocessed_data",
    output_directory="./results"
)

# Should complete without errors
assert not results.empty
```

---

## SUMMARY

**Critical Issues (Must Fix):**
1. ❌ String representation conversion for all list/tuple columns
2. ⚠️ Stanford tagger path handling with `os.path.join()`
3. ⚠️ Path validation before attempting to use Stanford tagger

**Nice-to-Have Improvements:**
- Progress bars for long operations
- Better error messages
- Input validation
- Column ordering
- Error recovery

**Estimated Effort:**
- Critical fixes: ~2 hours
- Nice-to-have improvements: ~4 hours
- Testing and validation: ~2 hours
- **Total: ~8 hours**

---

## NEXT STEPS

1. **Review this analysis** - Do you agree with the findings?
2. **Prioritize fixes** - Which issues are most critical for your workflow?
3. **Implement fixes** - I can provide the complete refactored code
4. **Test thoroughly** - Use the testing recommendations above
5. **Update documentation** - Reflect the new functionality in README

Would you like me to:
- A) Provide the complete refactored `prepare_transcripts.py` with all fixes?
- B) Create a minimal patch file with just the critical fixes?
- C) Move on to questions B-E about package integration?
