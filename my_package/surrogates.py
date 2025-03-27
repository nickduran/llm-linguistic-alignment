# my_package/surrogates.py
import os
import re
import random
import glob
import time
import pandas as pd
from itertools import combinations

class SurrogateGenerator:
    """
    Creates surrogate conversation pairs from real conversation data
    to establish baseline alignment levels.
    """
    
    def __init__(self, output_directory=None):
        """
        Initialize the surrogate generator
        
        Args:
            output_directory: Directory to save surrogate files (optional)
        """
        self.output_directory = output_directory or "surrogate_data"
    
    def generate_surrogates(self, 
                        original_conversation_list,
                        all_surrogates=True,
                        keep_original_turn_order=True,
                        id_separator='-',
                        dyad_label='dyad',
                        condition_label='cond'):
        """
        Create transcripts for surrogate pairs of participants who did not
        genuinely interact with each other.
        
        Args:
            original_conversation_list: List of paths to original conversation files
            all_surrogates: Whether to generate all possible surrogate pairings (default: True)
            keep_original_turn_order: Whether to maintain original turn order (default: True)
            id_separator: Character separating dyad ID from condition ID (default: '-')
            dyad_label: String preceding dyad ID in filenames (default: 'dyad')
            condition_label: String preceding condition ID in filenames (default: 'cond')
            
        Returns:
            list: Paths to all generated surrogate files
        """
        # Create a subfolder for the new set of surrogates
        new_surrogate_path = os.path.join(self.output_directory, f'surrogate_run-{time.time()}/')
        os.makedirs(new_surrogate_path, exist_ok=True)
        
        # Extract filenames without extensions
        file_info = [re.sub(r'\.txt','', os.path.basename(file_name)) for file_name in original_conversation_list]
        
        # Check filenames and filter to valid ones only
        valid_files = []
        problematic_files = []
        
        for i, file_name in enumerate(original_conversation_list):
            basename = os.path.basename(file_name)
            info = file_info[i]
            
            # Check if the filename has all required components
            has_dyad = dyad_label in info
            has_condition = condition_label in info
            has_separator = re.search(id_separator.strip("\\"), info) is not None
            
            if has_dyad and has_condition and has_separator:
                valid_files.append(file_name)
            else:
                problematic_files.append({
                    'file': basename,
                    'has_dyad': has_dyad,
                    'has_condition': has_condition,
                    'has_separator': has_separator
                })
        
        # Report on problematic files
        if problematic_files:
            print(f"WARNING: {len(problematic_files)} files have invalid naming patterns and will be skipped:")
            for i, file_info in enumerate(problematic_files[:5]):  # Show up to 5 examples
                print(f"  {i+1}. {file_info['file']} - Missing: " + 
                    (f"'{dyad_label}' " if not file_info['has_dyad'] else "") +
                    (f"'{condition_label}' " if not file_info['has_condition'] else "") +
                    (f"'{id_separator}' " if not file_info['has_separator'] else ""))
            if len(problematic_files) > 5:
                print(f"  ... and {len(problematic_files) - 5} more")
        
        # Check if we have enough valid files
        if len(valid_files) < 2:
            raise ValueError(f"Not enough valid files to generate surrogates. Found only {len(valid_files)} valid files.")
        
        print(f"Found {len(valid_files)} valid files for surrogate generation")
        
        # Group files by condition
        try:
            condition_pattern = f'[^{id_separator}]*{condition_label}[^{id_separator}]*'
            condition_ids = set()
            for file_name in valid_files:
                base_name = os.path.basename(file_name)
                matches = re.findall(condition_pattern, base_name)
                if matches:
                    condition_ids.add(matches[0])
        except Exception as e:
            print(f"Error extracting condition IDs: {e}")
            print(f"Pattern used: {condition_pattern}")
            print(f"Example filename: {os.path.basename(valid_files[0])}")
            raise ValueError("Could not extract condition IDs from filenames. Check your id_separator and condition_label parameters.")
        
        condition_ids = list(condition_ids)
        files_conditions = {}
        for unique_condition in condition_ids:
            next_condition_files = [add_file for add_file in valid_files 
                                if unique_condition in os.path.basename(add_file)]
            files_conditions[unique_condition] = next_condition_files
        
        # Generate surrogate pairs for each condition
        surrogate_files_created = []
        
        # Process each condition
        for condition in list(files_conditions.keys()):
            condition_files = files_conditions[condition]
            print(f"Processing condition '{condition}' with {len(condition_files)} files")
            
            # Skip if fewer than 2 files in this condition
            if len(condition_files) < 2:
                print(f"  Skipping condition '{condition}' - not enough files")
                continue
            
            # Get all possible pairings or a subset
            if all_surrogates:
                paired_surrogates = list(combinations(condition_files, 2))
            else:
                # Sample approximately half as many surrogate pairs as original conversations
                import math
                num_surrogates = int(math.ceil(len(condition_files)/2))
                all_possible_pairs = list(combinations(condition_files, 2))
                paired_surrogates = random.sample(all_possible_pairs, 
                                                min(num_surrogates, len(all_possible_pairs)))
            
            print(f"  Generating {len(paired_surrogates)} surrogate pairs")
            
            # Process each surrogate pairing
            for next_surrogate in paired_surrogates:
                try:
                    # Read original files
                    original_file1 = os.path.basename(next_surrogate[0])
                    original_file2 = os.path.basename(next_surrogate[1])
                    original_df1 = pd.read_csv(next_surrogate[0], sep='\t', encoding='utf-8')
                    original_df2 = pd.read_csv(next_surrogate[1], sep='\t', encoding='utf-8')
                    
                    # Validate dataframes
                    if len(original_df1) < 1 or len(original_df2) < 1:
                        print(f"  Skipping empty files: {original_file1} or {original_file2}")
                        continue
                    
                    # Extract dyad IDs
                    dyad_pattern = f'{dyad_label}[^{id_separator}]*'
                    try:
                        original_dyad1 = re.findall(dyad_pattern, original_file1)[0]
                        original_dyad2 = re.findall(dyad_pattern, original_file2)[0]
                    except IndexError:
                        print(f"  Could not extract dyad IDs from {original_file1} or {original_file2}")
                        print(f"  Pattern used: {dyad_pattern}")
                        continue
                    
                    # Check if participant column exists
                    if 'participant' not in original_df1.columns or 'participant' not in original_df2.columns:
                        print(f"  Missing 'participant' column in {original_file1} or {original_file2}")
                        continue
                    
                    # Get all unique participants from each file
                    participants1 = sorted(original_df1['participant'].unique())
                    participants2 = sorted(original_df2['participant'].unique())
                    
                    # Skip if either file has no participants
                    if len(participants1) == 0 or len(participants2) == 0:
                        print(f"  Skipping files with no participants: {original_file1} or {original_file2}")
                        continue
                    
                    # Determine how many participants to use (minimum of the two files)
                    num_participants = min(len(participants1), len(participants2))
                    print(f"  Found {len(participants1)} participants in file 1 and {len(participants2)} in file 2")
                    print(f"  Using {num_participants} participants for surrogate generation")
                    
                    # Create two surrogate conversation datasets
                    # First surrogate: Take participants from file1 and file2 alternating
                    surrogate_X_participants = []
                    for i in range(num_participants):
                        # Get participant from file 1
                        participant_df1 = original_df1[original_df1['participant'] == participants1[i]].reset_index()
                        participant_df1 = participant_df1.rename(columns={'file': 'original_file'})
                        
                        # Get participant from file 2
                        participant_df2 = original_df2[original_df2['participant'] == participants2[i]].reset_index()
                        participant_df2 = participant_df2.rename(columns={'file': 'original_file'})
                        
                        # Add to list of dataframes for this surrogate
                        surrogate_X_participants.append((participant_df1, participant_df2))
                    
                    # Second surrogate: Reverse the pairing
                    surrogate_Y_participants = []
                    for i in range(num_participants):
                        # Get participant from file 2
                        participant_df2 = original_df2[original_df2['participant'] == participants2[i]].reset_index()
                        participant_df2 = participant_df2.rename(columns={'file': 'original_file'})
                        
                        # Get participant from file 1  
                        participant_df1 = original_df1[original_df1['participant'] == participants1[i]].reset_index()
                        participant_df1 = participant_df1.rename(columns={'file': 'original_file'})
                        
                        # Add to list of dataframes for this surrogate
                        surrogate_Y_participants.append((participant_df2, participant_df1))
                    
                    # Process each surrogate conversation
                    for surrogate_idx, surrogate_participants in enumerate([surrogate_X_participants, surrogate_Y_participants]):
                        # Find minimum number of turns per participant
                        min_turns = float('inf')
                        for participant_pair in surrogate_participants:
                            min_turns = min(min_turns, len(participant_pair[0]), len(participant_pair[1]))
                        
                        if min_turns == float('inf') or min_turns == 0:
                            print(f"  Skipping surrogate {surrogate_idx+1}: no valid turns")
                            continue
                        
                        # Truncate each participant's turns to the minimum
                        truncated_participants = []
                        for participant_pair in surrogate_participants:
                            df1_trunc = participant_pair[0].truncate(after=min_turns-1, copy=True)
                            df2_trunc = participant_pair[1].truncate(after=min_turns-1, copy=True)
                            
                            # Optionally shuffle turns if requested
                            if not keep_original_turn_order:
                                df1_trunc = df1_trunc.sample(frac=1).reset_index(drop=True)
                                df2_trunc = df2_trunc.sample(frac=1).reset_index(drop=True)
                            
                            truncated_participants.append((df1_trunc, df2_trunc))
                        
                        # Combine all participants' turns
                        all_dfs = []
                        for pair in truncated_participants:
                            all_dfs.extend([pair[0], pair[1]])
                        
                        # Sort by index to interleave participants' turns
                        surrogate_df = pd.concat(all_dfs).sort_index(kind="mergesort").reset_index(drop=True)
                        surrogate_df = surrogate_df.rename(columns={'index': 'original_index'})
                        
                        # Name the surrogate file
                        if surrogate_idx == 0:
                            # First surrogate: participants from file1 and file2
                            surrogate_df['file'] = f"{original_dyad1}-{original_dyad2}-{condition}"
                            surrogate_name = f"SurrogatePair-{original_dyad1}A-{original_dyad2}B-{condition}.txt"
                        else:
                            # Second surrogate: participants from file2 and file1
                            surrogate_df['file'] = f"{original_dyad2}-{original_dyad1}-{condition}"
                            surrogate_name = f"SurrogatePair-{original_dyad2}A-{original_dyad1}B-{condition}.txt"
                        
                        # Save surrogate file
                        surrogate_path = os.path.join(new_surrogate_path, surrogate_name)
                        surrogate_df.to_csv(surrogate_path, encoding='utf-8', index=False, sep='\t')
                        surrogate_files_created.append(surrogate_path)
                        print(f"  Created surrogate file: {surrogate_name}")
                    
                except Exception as e:
                    print(f"  Error processing surrogate pair: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        num_created = len(surrogate_files_created)
        if num_created > 0:
            print(f"Successfully created {num_created} surrogate files")
            return surrogate_files_created
        else:
            raise ValueError("Failed to create any surrogate files. Check the logs for details.")


class SurrogateAlignment:
    """
    Calculates alignment metrics for surrogate pairs to establish baseline levels
    of alignment that would occur by chance.
    """
    
    def __init__(self, alignment_type="lexsyn", cache_dir=None, model_name=None, token=None):
        """
        Initialize the surrogate alignment analyzer
        
        Args:
            alignment_type: Type of alignment to use for analysis
            cache_dir: Directory to cache models (optional)
            model_name: Name of the specific model to use (optional)
            token: API token for model access (optional)
        """
        from .alignment import LinguisticAlignment
        self.alignment = LinguisticAlignment(
            alignment_type=alignment_type,  # Changed from embedding_model
            model_name=model_name,
            token=token,
            cache_dir=cache_dir
        )
        self.surrogate_generator = SurrogateGenerator()
    
    def analyze_baseline(self, input_files, output_directory="results", surrogate_directory=None,
                        all_surrogates=True, keep_original_turn_order=True, id_separator='-',
                        condition_label='cond', dyad_label='dyad', lag=1, max_ngram=2,
                        high_sd_cutoff=3, low_n_cutoff=1, save_vocab=False,
                        ignore_duplicates=True, add_stanford_tags=False, **kwargs):
        """
        Generate surrogate pairs and analyze their alignment as a baseline
        """
        # Ensure root output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Set up surrogate directory
        surrogate_dir = surrogate_directory or os.path.join(output_directory, "surrogates")
        os.makedirs(surrogate_dir, exist_ok=True)
        
        # Get cache directory from the alignment object
        cache_dir = None
        if hasattr(self.alignment, 'w2v_wrapper') and hasattr(self.alignment.w2v_wrapper, 'cache_dir'):
            cache_dir = self.alignment.w2v_wrapper.cache_dir
        elif hasattr(self.alignment.analyzer, 'cache_dir'):
            cache_dir = self.alignment.analyzer.cache_dir
        
        # Resolve input files
        if isinstance(input_files, str) and os.path.isdir(input_files):
            file_list = glob.glob(os.path.join(input_files, "*.txt"))
        elif isinstance(input_files, list):
            file_list = input_files
        else:
            raise ValueError("input_files must be a directory path or list of file paths")
        
        # Generate surrogate conversation pairs
        print(f"Generating surrogate conversation pairs from {len(file_list)} files...")
        self.surrogate_generator.output_directory = surrogate_dir
        surrogate_files = self.surrogate_generator.generate_surrogates(
            original_conversation_list=file_list,
            all_surrogates=all_surrogates,
            keep_original_turn_order=keep_original_turn_order,
            id_separator=id_separator,
            dyad_label=dyad_label,
            condition_label=condition_label
        )
        
        print(f"Generated {len(surrogate_files)} surrogate files")
        
        # Run alignment analysis on surrogate pairs
        print("Running alignment analysis on surrogate pairs...")
        
        # Create a dictionary of parameters to pass
        alignment_params = {
            'folder_path': os.path.dirname(surrogate_files[0]),
            'output_directory': None,  # Don't save intermediate results
            'file_pattern': "*.txt",
            'lag': lag,
            'max_ngram': max_ngram,
            'ignore_duplicates': ignore_duplicates,
            'add_stanford_tags': add_stanford_tags,
            'save_vocab': False,  # Don't save vocabulary again
            'high_sd_cutoff': high_sd_cutoff,
            'low_n_cutoff': low_n_cutoff
        }
        
        # Add cache_dir if it exists
        if cache_dir:
            alignment_params['cache_dir'] = cache_dir
        
        # Add any additional kwargs
        alignment_params.update(kwargs)
        
        # Run the analysis
        surrogate_results = self.alignment.analyze_folder(**alignment_params)
        
        # Save results with baseline-specific filename, directly in the main directory
        if not surrogate_results.empty:
            # Remove embedding columns
            columns_to_remove = [col for col in surrogate_results.columns if 'embedding' in col or '_dims' in col]
            surrogate_results_clean = surrogate_results.drop(columns=columns_to_remove, errors='ignore')
            
            # Create baseline-specific filename
            alignment_type = self.alignment.alignment_type
            baseline_path = os.path.join(output_directory, f"baseline_alignment_{alignment_type}_lag{lag}.csv")
            
            # Add model-specific parameters to filename
            if alignment_type == "lexsyn":
                dup_str = "noDups" if ignore_duplicates else "withDups"
                stan_str = "withStan" if add_stanford_tags else "noStan"
                baseline_path = os.path.join(
                    output_directory, 
                    f"baseline_alignment_lexsyn_ngram{max_ngram}_lag{lag}_{dup_str}_{stan_str}.csv"
                )
            elif alignment_type == "fasttext":
                sd_str = f"sd{high_sd_cutoff}"
                n_str = f"n{low_n_cutoff}"
                baseline_path = os.path.join(
                    output_directory, 
                    f"baseline_alignment_fasttext_lag{lag}_{sd_str}_{n_str}.csv"
                )
            
            # Save the results
            surrogate_results_clean.to_csv(baseline_path, index=False)
            print(f"Baseline alignment results saved to {baseline_path}")
        
        return surrogate_results