import os
import torch
from sentence_transformers import SentenceTransformer

CPU = 1
if CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Save the model's config and state_dict
model_save_path = 'all-MiniLM-L6-v2/'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")