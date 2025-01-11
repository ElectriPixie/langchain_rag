import os
import json
import torch
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

import argparse

parser = argparse.ArgumentParser()

# Define the arguments
parser.add_argument('--vstoreName', type=str, default='Book_Collection')
parser.add_argument('--vstoreDir', type=str, default='faiss_store/')
parser.add_argument('--modelPath', type=str, default='all-MiniLM-L6-v2/')
parser.add_argument("--cpu", choices=["True", "False"], default="False")
# Parse the arguments
args = parser.parse_args()

def add_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path

# Assign the values to the variables
vstoreName = add_trailing_slash(args.vstoreName)
vstoreDir = add_trailing_slash(args.vstoreDir)
modelPath = args.modelPath
vstorePath = vstoreDir+vstoreName
cpu = args.cpu

if cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.makedirs(vstorePath, exist_ok=True)

# Define the custom embeddings class that inherits from LangChain's Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, modelPath: str):
        self.model = SentenceTransformer(modelPath)

    def embed_query(self, query: str):
        if cpu: 
            return self.model.encode([query], convert_to_tensor=True)[0].cpu().numpy()
        else:
            return self.model.encode([query], convert_to_tensor=True)[0].numpy()

    def embed_documents(self, documents: list):
        return [self.embed_query(doc) for doc in documents]

# Set default device to CPU
torch.set_default_device('cpu')

# Create custom embeddings object
embeddings = SentenceTransformerEmbeddings(modelPath=modelPath)

# Create the FAISS index
index = faiss.IndexFlatL2(embeddings.model.get_sentence_embedding_dimension())

# Create the FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,  # Pass the custom embeddings object
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Save the FAISS index
faiss.write_index(index, os.path.join(vstorePath, f"index.faiss"))

# Create and save an empty document store (No documents added yet)
with open(os.path.join(vstorePath, f"documents.json"), "w") as doc_file:
    json.dump({}, doc_file)

print(f"FAISS index and embeddings initialized and saved to {vstorePath}. No documents added.")
