import torch
from sentence_transformers import SentenceTransformer

# Load the pre-trained model from the Hugging Face Hub
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Save the model's state dictionary
torch.save(model.state_dict(), 'all-MiniLM-L6-v2.pt')
print("Model saved as all-MiniLM-L6-v2.pt")

