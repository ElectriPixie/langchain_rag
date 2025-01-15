import os
import torch
from sentence_transformers import SentenceTransformer
import psutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'sharedFunctions'))
from sharedFunctions import add_trailing_slash
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from config import DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME
import argparse

def get_program_name():
    parent_pid = os.getppid()
    parent_process = psutil.Process(parent_pid)
    parent_cmdline = parent_process.cmdline()

    if len(parent_cmdline) > 1:  # Check if there are arguments
        run_script_name = os.path.basename(parent_cmdline[1])  # Get only the file name
    else:
        run_script_name = "Unknown"

    script_name = os.path.basename(__file__)

    run_script_base = os.path.splitext(run_script_name)[0]
    script_base = os.path.splitext(script_name)[0]

    if run_script_base == script_base:
        return run_script_name
    else:
        return script_name

def main():
    SCRIPT_DIR = os.path.dirname(__file__)
    DEFAULT_PATH = add_trailing_slash(os.path.dirname(SCRIPT_DIR))

    prog_name = get_program_name()

    parser = argparse.ArgumentParser(prog=prog_name)

    parser.add_argument('--modelDir',
                        type=str,
                        default=DEFAULT_MODEL_DIR,
                        help='Model dir: The directory to store models.')

    parser.add_argument('--modelName', 
                        type=str, 
                        default=DEFAULT_MODEL_NAME, 
                        help='Model name: The name of the model to be used. This is used to load the model. (e.g. "all-MiniLM-L6-v2")')

    parser.add_argument("--gpu", 
                        action='store_true', 
                        help='Device: Run on GPU instead of CPU. (Default: CPU)')

    args = parser.parse_args()

    modelDir = args.modelDir
    modelName = args.modelName
    gpu = args.gpu

    if modelDir is not DEFAULT_MODEL_DIR:
        modelDir = add_trailing_slash(modelDir)
    else:
        modelDir = add_trailing_slash(DEFAULT_PATH+modelDir)

    modelPath = modelDir+modelName
    print(modelPath)

    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = SentenceTransformer(modelName)
    model.save(modelPath)
    print(f"Model saved to {modelPath}")

if __name__ == "__main__":
    main()