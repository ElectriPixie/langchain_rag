import os
import torch
from sentence_transformers import SentenceTransformer
import argparse

# Define the argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2')
parser.add_argument("--cpu", choices=["True", "False"], default="False")
args = parser.parse_args()
cpu = args.cpu


if cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the pre-trained model
model = SentenceTransformer(args.model_name)

# Save the model's config and state_dict
model_save_path = args.model_name
model.save(model_save_path)
print(f"Model saved to {model_save_path}")