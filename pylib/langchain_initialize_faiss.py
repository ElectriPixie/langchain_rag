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

def add_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_PATH = add_trailing_slash(os.path.dirname(SCRIPT_DIR))
DEFAULT_VSTORE_NAME="Book_Collection"
DEFAULT_VSTORE_DIR="faiss_store"
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

parser = argparse.ArgumentParser()
# Define the name of the vector store
parser.add_argument('--vstoreName',
                    type=str,
                    default=DEFAULT_VSTORE_NAME,
                    help='Vector store name: The name of the vector store. This is used to identify the vector store.')

# Define the directory where the vector store is located
parser.add_argument('--vstoreDir',
                    type=str,
                    default=DEFAULT_VSTORE_DIR,
                    help='Vector store directory: The directory where the vector store is located.')

parser.add_argument('--modelDir',
                    type=str,
                    default=DEFAULT_MODEL_DIR,
                    help='Model dir: The directory to store models. This is used to load the model')

parser.add_argument('--modelName', 
                    type=str, 
                    default=DEFAULT_MODEL_NAME, 
                    help='Model name: The name of the model to be used. This is used to load the model. (e.g. "all-MiniLM-L6-v2")')

# Define the device to use (CPU or GPU)
parser.add_argument('--gpu',
                    action='store_true',
                    help='Device: Use GPU instead of CPU (default). This is used to specify the device to use.')

# Parse the arguments
args = parser.parse_args()

if args.vstoreDir is not DEFAULT_VSTORE_DIR:
    vstoreDir = add_trailing_slash(args.vstoreDir)
else:
   vstoreDir = add_trailing_slash(DEFAULT_PATH+args.vstoreDir)

if args.modelDir is not DEFAULT_MODEL_DIR:
    modelDir = add_trailing_slash(args.modelDir)
else:
   modelDir = add_trailing_slash(DEFAULT_PATH+args.modelDir)

modelName = add_trailing_slash(args.modelName)
vstoreName = add_trailing_slash(args.vstoreName)
gpu = args.gpu

modelPath = modelDir+modelName
vstorePath=vstoreDir+vstoreName

if not gpu:
    # Set default device to CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_device('cpu')

os.makedirs(vstorePath, exist_ok=True)

# Define the custom embeddings class that inherits from LangChain's Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, modelPath: str):
        self.model = SentenceTransformer(modelPath)

    def embed_query(self, query: str):
        if gpu: 
            return self.model.encode([query], convert_to_tensor=True)[0].numpy()
        else:
            return self.model.encode([query], convert_to_tensor=True)[0].cpu().numpy()

    def embed_documents(self, documents: list):
        return [self.embed_query(doc) for doc in documents]


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
