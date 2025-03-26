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
                           id_separator='\-',
                           dyad_label='dyad',
                           condition_label='cond'):
        """
        Create transcripts for surrogate pairs of participants who did not
        genuinely interact with each other.
        
        Args:
            original_conversation_list: List of paths to original conversation files
            all_surrogates: Whether to generate all possible surrogate pairings (default: True)
            keep_original_turn_order: Whether to maintain original turn order (default: True)
            id_separator: Character separating dyad ID from condition ID (default: '\-')
            dyad_label: String preceding dyad ID in filenames (default: 'dyad')
            condition_label: String preceding condition ID in filenames (default: 'cond')
            
        Returns:
            list: Paths to all generated surrogate files
        """
        # Create a subfolder for the new set of surrogates
        new_surrogate_path = os.path.join(self.output_directory, f'surrogate_run-{time.time()}/')
        os.makedirs(new_surrogate_path, exist_ok=True)
        
        # Extract filenames without extensions
        file_info = [re.sub('\.txt','', os.path.basename(file_name)) for file_name in original_conversation_list]
        
        # Validate filenames
        if not all(dyad_label in name and condition_label in name and re.search(id_separator.strip("\\"), name) 
                  for name in file_info):
            raise ValueError(f"Filenames must include '{dyad_label}', '{condition_label}', and '{id_separator}'")
        
        # Group files by condition
        condition_ids = list(set([re.findall(f'[^{id_separator}]*{condition_label}.*', metadata)[0] 
                                for metadata in file_info]))
        files_conditions = {}
        for unique_condition in condition_ids:
            next_condition_files = [add_file for add_file in original_conversation_list 
                                   if unique_condition in add_file]
            files_conditions[unique_condition] = next_condition_files
        
        # Process each condition
        for condition in list(files_conditions.keys()):
            # Get all possible pairings or a subset
            if all_surrogates:
                paired_surrogates = list(combinations(files_conditions[condition], 2))
            else:
                # Sample approximately half as many surrogate pairs as original conversations
                import math
                num_surrogates = int(math.ceil(len(files_conditions[condition])/2))
                all_possible_pairs = list(combinations(files_conditions[condition], 2))
                paired_surrogates = random.sample(all_possible_pairs, 
                                                 min(num_surrogates, len(all_possible_pairs)))
            
            # Process each surrogate pairing
            for next_surrogate in paired_surrogates:
                # Read original files
                original_file1 = os.path.basename(next_surrogate[0])
                original_file2 = os.path.basename(next_surrogate[1])
                original_df1 = pd.read_csv(next_surrogate[0], sep='\t', encoding='utf-8')
                original_df2 = pd.read_csv(next_surrogate[1], sep='\t', encoding='utf-8')
                
                # Validate dataframes
                if len(original_df1) < 1 or len(original_df2) < 1:
                    print(f"Skipping empty files: {original_file1} or {original_file2}")
                    continue
                
                # Get participants from df1
                participantA_1_code = min(original_df1['participant'].unique())
                participantB_1_code = max(original_df1['participant'].unique())
                participantA_1 = original_df1[original_df1['participant'] == participantA_1_code].reset_index()
                participantA_1 = participantA_1.rename(columns={'file': 'original_file'})
                participantB_1 = original_df1[original_df1['participant'] == participantB_1_code].reset_index()
                participantB_1 = participantB_1.rename(columns={'file': 'original_file'})
                
                # Get participants from df2
                participantA_2_code = min(original_df2['participant'].unique())
                participantB_2_code = max(original_df2['participant'].unique())
                participantA_2 = original_df2[original_df2['participant'] == participantA_2_code].reset_index()
                participantA_2 = participantA_2.rename(columns={'file': 'original_file'})
                participantB_2 = original_df2[original_df2['participant'] == participantB_2_code].reset_index()
                participantB_2 = participantB_2.rename(columns={'file': 'original_file'})
                
                # Determine truncation points
                surrogateX_turns = min(len(participantA_1), len(participantB_2))
                surrogateY_turns = min(len(participantA_2), len(participantB_1))
                
                # Create surrogate pairs
                if keep_original_turn_order:
                    # Preserve original turn ordering
                    surrogateX_A1 = participantA_1.truncate(after=surrogateX_turns-1, copy=True)
                    surrogateX_B2 = participantB_2.truncate(after=surrogateX_turns-1, copy=True)
                    surrogateX = pd.concat([surrogateX_A1, surrogateX_B2]).sort_index(
                        kind="mergesort").reset_index(drop=True).rename(
                        columns={'index': 'original_index'})
                    
                    surrogateY_A2 = participantA_2.truncate(after=surrogateY_turns-1, copy=True)
                    surrogateY_B1 = participantB_1.truncate(after=surrogateY_turns-1, copy=True)
                    surrogateY = pd.concat([surrogateY_A2, surrogateY_B1]).sort_index(
                        kind="mergesort").reset_index(drop=True).rename(
                        columns={'index': 'original_index'})
                else:
                    # Shuffle turns within participants
                    surrogateX_A1 = participantA_1.truncate(after=surrogateX_turns-1, copy=True).sample(frac=1).reset_index(drop=True)
                    surrogateX_B2 = participantB_2.truncate(after=surrogateX_turns-1, copy=True).sample(frac=1).reset_index(drop=True)
                    surrogateX = pd.concat([surrogateX_A1, surrogateX_B2]).sort_index(
                        kind="mergesort").reset_index(drop=True).rename(
                        columns={'index': 'original_index'})
                    
                    surrogateY_A2 = participantA_2.truncate(after=surrogateY_turns-1, copy=True).sample(frac=1).reset_index(drop=True)
                    surrogateY_B1 = participantB_1.truncate(after=surrogateY_turns-1, copy=True).sample(frac=1).reset_index(drop=True)
                    surrogateY = pd.concat([surrogateY_A2, surrogateY_B1]).sort_index(
                        kind="mergesort").reset_index(drop=True).rename(
                        columns={'index': 'original_index'})
                
                # Set filenames for surrogate pairs
                original_dyad1 = re.findall(f'{dyad_label}[^{id_separator}]*', original_file1)[0]
                original_dyad2 = re.findall(f'{dyad_label}[^{id_separator}]*', original_file2)[0]
                surrogateX['file'] = f"{original_dyad1}-{original_dyad2}-{condition}"
                surrogateY['file'] = f"{original_dyad2}-{original_dyad1}-{condition}"
                nameX = f"SurrogatePair-{original_dyad1}A-{original_dyad2}B-{condition}.txt"
                nameY = f"SurrogatePair-{original_dyad2}A-{original_dyad1}B-{condition}.txt"
                
                # Save surrogate files
                surrogateX.to_csv(os.path.join(new_surrogate_path, nameX), encoding='utf-8', index=False, sep='\t')
                surrogateY.to_csv(os.path.join(new_surrogate_path, nameY), encoding='utf-8', index=False, sep='\t')
                
                print(f"Created surrogate pairs: {nameX} and {nameY}")
        
        # Return list of all surrogate files
        return glob.glob(os.path.join(new_surrogate_path, "*.txt"))


class SurrogateAlignment:
    """
    Calculates alignment metrics for surrogate pairs to establish baseline levels
    of alignment that would occur by chance.
    """
    
    def __init__(self, embedding_model="lexsyn", cache_dir=None):
        """
        Initialize the surrogate alignment analyzer
        
        Args:
            embedding_model: Type of embedding model to use for alignment
            cache_dir: Directory to cache models (optional)
        """
        from .alignment import SemanticAlignment
        self.alignment = SemanticAlignment(embedding_model=embedding_model, cache_dir=cache_dir)
        self.surrogate_generator = SurrogateGenerator()
    
    def analyze_baseline(self, 
                            input_files,
                            output_directory="results",
                            surrogate_directory=None,
                            all_surrogates=True,
                            keep_original_turn_order=True,
                            id_separator='\-',
                            condition_label='cond',
                            dyad_label='dyad',
                            lag=1,
                            max_ngram=2,
                            ignore_duplicates=True,
                            add_stanford_tags=False,
                            **kwargs):
            """
            Generate surrogate pairs and analyze their alignment as a baseline
            
            Args:
                input_files: Path to directory containing conversation files or list of file paths
                output_directory: Directory to save alignment results
                surrogate_directory: Directory to save surrogate files (optional)
                all_surrogates: Whether to generate all possible surrogate pairings (default: True)
                keep_original_turn_order: Whether to maintain original turn order (default: True)
                id_separator: Character separating dyad ID from condition ID (default: '\-')
                condition_label: String preceding condition ID in filenames (default: 'cond')
                dyad_label: String preceding dyad ID in filenames (default: 'dyad')
                lag: Number of turns to lag when analyzing alignment (default: 1)
                max_ngram: Maximum n-gram size for lexical/syntactic analysis (default: 2)
                ignore_duplicates: Whether to ignore duplicate n-grams (default: True)
                add_stanford_tags: Whether to include Stanford POS tags (default: False)
                **kwargs: Additional arguments for alignment analysis
                
            Returns:
                pd.DataFrame: Alignment results for surrogate pairs
            """
            # Ensure output directories exist
            os.makedirs(output_directory, exist_ok=True)
            surrogate_dir = surrogate_directory or os.path.join(output_directory, "surrogates")
            os.makedirs(surrogate_dir, exist_ok=True)
            
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
            surrogate_results = self.alignment.analyze_folder(
                folder_path=os.path.dirname(surrogate_files[0]),  # Use the directory where surrogates were saved
                output_directory=os.path.join(output_directory, "baseline"),
                file_pattern="*.txt",
                lag=lag,
                max_ngram=max_ngram,
                ignore_duplicates=ignore_duplicates,
                add_stanford_tags=add_stanford_tags,
                **kwargs
            )
            
            # Save results with special filename to indicate these are baseline/surrogate results
            if not surrogate_results.empty:
                # Create baseline-specific filename
                dup_str = "noDups" if ignore_duplicates else "withDups"
                stan_str = "withStan" if add_stanford_tags else "noStan"
                baseline_path = os.path.join(
                    output_directory, 
                    f"baseline_alignment_{self.alignment.embedding_model}_ngram{max_ngram}_lag{lag}_{dup_str}_{stan_str}.csv"
                )
                
                # Save the results
                surrogate_results.to_csv(baseline_path, index=False)
                print(f"Baseline alignment results saved to {baseline_path}")
            
            return surrogate_results