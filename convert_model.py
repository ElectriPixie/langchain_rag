import os
import torch
from sentence_transformers import SentenceTransformer
import argparse
import psutil

import psutil

# Get the parent process ID
parent_pid = os.getppid()

# Fetch the parent process
parent_process = psutil.Process(parent_pid)

# Use cmdline to get the full command with arguments
parent_cmdline = parent_process.cmdline()

# Extract the calling script name from cmdline
if len(parent_cmdline) > 1:  # Check if there are arguments
    run_script_name = os.path.basename(parent_cmdline[1])  # Get only the file name
else:
    run_script_name = "Unknown"

# Get the current Python script name
script_name = os.path.basename(__file__)

# Remove extensions for comparison
run_script_base = os.path.splitext(run_script_name)[0]
script_base = os.path.splitext(script_name)[0]

# Compare base names
if run_script_base == script_base:
    prog_name=run_script_name
else:
    prog_name=script_name

# Define the argparse
parser = argparse.ArgumentParser(prog=prog_name)

# Add a command-line argument for the model name
parser.add_argument('--modelName', 
                    type=str, 
                    default='all-MiniLM-L6-v2', 
                    help='The name of the model to be used. (e.g. "all-MiniLM-L6-v2")')

# Add a command-line flag for the CPU usage
parser.add_argument("--cpu", 
                    action='store_true', 
                    help='Run on CPU')

args = parser.parse_args()
modelName = args.modelName
cpu = args.cpu

if cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the pre-trained model
model = SentenceTransformer(args.modelName)

# Save the model's config and state_dict
model_save_path = modelName
model.save(model_save_path)
print(f"Model saved to {model_save_path}")