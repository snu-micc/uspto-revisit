import json
import os
import pickle
import time
import pandas as pd
from json_name_fix_utils import is_valid_json, fix_json_string, fix_name

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_smiles_dict(smiles_dict, filename):
    with open(filename, 'w', encoding='utf-8-sig') as f:
        json.dump(smiles_dict, f, ensure_ascii=False, indent=2)

# Function to monitor log in real-time
def monitor_log(file_path, stop_event, lines_per_batch=1):
    with open(file_path, 'r') as f:
        f.seek(0, 2)  # Go to the end of the file
        batch_counter = 0
        line_counter = 0

        while not stop_event.is_set():
            line = f.readline()
            if line:
                if "Starting batch" in line:  # Reset line counter for each new batch
                    line_counter = 0
                    batch_counter += 1
                    print(f"--- Batch {batch_counter} ---")
                
                if line_counter < lines_per_batch:
                    print(line, end='')
                    line_counter += 1
                
                if "Processing completed" in line:
                    print("All batches processed. Stopping log monitoring.")
                    stop_event.set() 
            else:
                time.sleep(0.1)  # Wait briefly before checking again

def add_smiles_dict(column_name, exisitng_file_path, smiles_dict_file_path, output_path):
    GPT_response = pd.read_csv(exisitng_file_path)
    with open(smiles_dict_file_path, 'r', encoding='utf-8-sig') as f:
        smiles_dict = json.load(f)

    GPT_response[column_name] = None
    for idx, row in GPT_response.iterrows():
        GPT_response.at[idx, column_name] = smiles_dict[idx]


    # Save the updated DataFrame with the new column
    GPT_response.to_csv(output_path, index=False, encoding='utf-8-sig')
    

