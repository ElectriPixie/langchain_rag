import os
import json
import torch
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import argparse
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

parser = argparse.ArgumentParser(prog=prog_name)

# Define the name of the vector store
parser.add_argument('--vstoreName',
                    type=str,
                    default='Book_Collection',
                    help='Vector store name: The name of the vector store. This is used to identify the vector store.')

# Define the directory where the vector store is located
parser.add_argument('--vstoreDir',
                    type=str,
                    default='faiss_store/',
                    help='Vector store directory: The directory where the vector store is located.')

# Define the path to the model to be used
parser.add_argument('--modelPath',
                    type=str,
                    default='all-MiniLM-L6-v2/',
                    help='Model path: The path to the model to be used. This is used to load the model.')

# Define the device to use (CPU or GPU)
parser.add_argument('--cpu',
                    action='store_true',
                    help='Device: Use CPU instead of GPU (default). This is used to specify the device to use.')

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
