import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import torch
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
import json
import argparse
import sys
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


def add_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path

# Argument Parser
parser = argparse.ArgumentParser(prog=prog_name)

def print_help_and_exit():
    parser.print_help()
    sys.exit(0)

# Define the FAISS store name
parser.add_argument('--vstoreName',
                    type=str,
                    default='Book_Collection',
                    help='Vector store name: The name of the vector store. This is used to identify the vector store.')

# Specify the directory to store the FAISS index
parser.add_argument('--vstoreDir',
                    type=str,
                    default='faiss_store/',
                    help='Vector store directory: The directory where the vector store is located.')

# Define the directory containing PDF files
parser.add_argument('--pdfDir',
                    type=str,
                    default='pdf',
                    help='PDF directory: The directory containing PDF files.')

# Specify the path to load the model
parser.add_argument('--modelPath',
                    type=str,
                    default='all-MiniLM-L6-v2/',
                    help='Model path: The path to the model to be used. This is used to load the model.')

# Run on CPU
parser.add_argument('--cpu',
                    action='store_true',
                    help='Device: Use CPU instead of GPU (default). This is used to specify the device to use.')

# Use per-page embeddings
parser.add_argument('--perPageEmbeddings',
                    choices=["True", "False"],
                    default="False",
                    help='Per-page embeddings: Specify whether to use per-page embeddings. This can improve the accuracy of the model.')
args = parser.parse_args()

# Assign the values to the variables
vstoreName = add_trailing_slash(args.vstoreName)
vstoreDir = add_trailing_slash(args.vstoreDir)
vstorePath = vstoreDir+vstoreName
pdfDir = add_trailing_slash(args.pdfDir)
modelPath = args.modelPath
perPageEmbeddings = args.perPageEmbeddings
cpu = args.cpu

if cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

# Function to extract text content from a Document object
def document_to_dict(doc):
# Converts a Document object to a dictionary for serialization.
    # Check for possible attributes that might hold the text
    if hasattr(doc, 'text'):
        return {'text': doc.text, 'metadata': doc.metadata}
    elif hasattr(doc, 'content'):
        return {'text': doc.content, 'metadata': doc.metadata}
    elif hasattr(doc, 'page_content'):
        return {'text': doc.page_content, 'metadata': doc.metadata}
    elif hasattr(doc, 'raw_text'):
        return {'text': doc.raw_text, 'metadata': doc.metadata}
    else:
        # If no known attributes are found, print the document for inspection
        print(f"Document structure: {doc}")
        return {'text': str(doc), 'metadata': None}  # Fallback to string conversion

def get_page_text(page):
    if hasattr(page, 'text'):
        return page.text
    elif hasattr(page, 'content'):
        return page.content
    elif hasattr(page, 'page_content'):
        return page.page_content
    elif hasattr(page, 'raw_text'):
        return page.raw_text
    else:
        print(f"Document structure: {page}")
        return str(page)

def load_existing_uuids_from_json(json_file):
    #Load UUIDs from the documents JSON file for fast lookup
    json_file = vstorePath+json_file
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as file:
                # Load the JSON data
                data = json.load(file)

                # Ensure data is a dictionary where the keys are UUIDs
                if isinstance(data, dict):
                    # Extract the UUIDs from the keys of the dictionary
                    return set(data.keys())
                else:
                    raise ValueError(f"Invalid JSON structure in {json_file}: Expected a dictionary with UUID keys")
        except (json.JSONDecodeError, ValueError) as e:
            # Handle invalid JSON format or structure gracefully
            print(f"Error loading JSON file {json_file}: {e}")
            return set()  # Return an empty set in case of error
    return set()  # Return an empty set if the file doesn't exist

def add_documents_to_store(pdfDir, json_file='documents.json'):
    all_documents = [] # List to store documents with their UUIDs

    if not os.path.exists(pdfDir):
        print("Error: PDF directory not found.")
        print_help_and_exit()
    existing_uuids = load_existing_uuids_from_json(json_file)

    for file in os.listdir(pdfDir):
        if file.endswith('.pdf'):
            pdfName = file

            file_path = os.path.join(pdfDir, file)
            loader = PyPDFLoader(file_path)
            pages = list(loader.lazy_load())  # Get the pages from the PDF file
            uuids = [file.replace('.pdf', '').replace('/', '_') + f"_{i}" for i, _ in enumerate(pages)]
            if uuids[0] in existing_uuids:
                print(f"Skipping: {pdfName}")
            else:
                print(f"Loading: {pdfName}")
                if perPageEmbeddings:
                    # Add documents to FAISS index
                    vector_store.add_documents(documents=pages, ids=uuids)
                else:
                    whole_document = '\n'.join(pages)  # Join the pages into a single document
                    # Add document to FAISS index
                    vector_store.add_document(document=whole_document, id=uuids[0])

                # Store documents and their UUIDs for saving
                for uuid, page in zip(uuids, pages):
                    all_documents.append({"uuid": uuid, "content": page})
    return all_documents


# Add documents to the vector store and retrieve document content
all_documents = add_documents_to_store(pdfDir)
if all_documents:
    # Save the FAISS index directly
    faiss.write_index(index, os.path.join(vstorePath, f"index.faiss"))

    # Save the document store (in memory)
    docstore_data = {doc["uuid"]: document_to_dict(doc["content"]) for doc in all_documents}

    with open(os.path.join(vstorePath, f"documents.json"), "w") as doc_file:
        json.dump(docstore_data, doc_file)

    print(f"Documents and FAISS index saved to {vstorePath}")
else:
    print(f"No Documents saved to {vstorePath}")