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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'sharedFunctions'))
from sharedFunctions import add_trailing_slash
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from config import DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME, DEFAULT_VSTORE_DIR, DEFAULT_VSTORE_NAME

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

    vstoreName = add_trailing_slash(args.vstoreName)
    modelName = add_trailing_slash(args.modelName)
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
    dimension = embeddings.model.get_sentence_embedding_dimension()
    index = faiss.index_factory(dimension, "Flat", faiss.METRIC_L2)

    # Save the FAISS index
    faiss.write_index(index, os.path.join(vstorePath, f"index.faiss"))

    # Create and save an empty document store (No documents added yet)
    with open(os.path.join(vstorePath, f"documents.json"), "w") as doc_file:
        json.dump({}, doc_file)

    print(f"FAISS index and embeddings initialized and saved to {vstorePath}. No documents added.")

if __name__ == "__main__":
    main()