import os
import torch
from sentence_transformers import SentenceTransformer
import argparse
import psutil

def add_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_PATH = add_trailing_slash(os.path.dirname(SCRIPT_DIR))
DEFAULT_MODEL_DIR="models"
DEFAULT_MODEL_NAME="all-MiniLM-L6-v2"

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
parser.add_argument('--modelDir',
                    type=str,
                    default=DEFAULT_MODEL_DIR,
                    help='Model dir: The directory to store models.')

parser.add_argument('--modelName', 
                    type=str, 
                    default=DEFAULT_MODEL_NAME, 
                    help='Model name: The name of the model to be used. This is used to load the model. (e.g. "all-MiniLM-L6-v2")')

# Add a command-line flag for the CPU usage
parser.add_argument("--gpu", 
                    action='store_true', 
                    help='Device: Run on GPU instead of CPU. (Default: CPU)')


args = parser.parse_args()

if args.modelDir is not DEFAULT_MODEL_NAME:
    modelDir = add_trailing_slash(args.modelDir)
else:
   modelDir = add_trailing_slash(DEFAULT_PATH+args.modelDir)

if args.modelName is not DEFAULT_MODEL_NAME:
    modelName = add_trailing_slash(args.modelName)
else:
   modelName = add_trailing_slash(DEFAULT_PATH+args.modelName)
modelPath = modelDir+modelName
gpu = args.gpu

if not gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the pre-trained model
model = SentenceTransformer(modelName)

# Save the model's config and state_dict
model.save(modelPath)
print(f"Model saved to {modelPath}")