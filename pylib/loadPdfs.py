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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'sharedFunctions'))
from sharedFunctions import add_trailing_slash, get_program_name, print_help_and_exit
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from config import DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME, DEFAULT_VSTORE_DIR, DEFAULT_VSTORE_NAME, DEFAULT_PDF_DIR

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_PATH = add_trailing_slash(os.path.dirname(SCRIPT_DIR))

prog_name = get_program_name()

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

# Define the directory containing PDF files
parser.add_argument('--pdfDir',
                    type=str,
                    default=DEFAULT_PDF_DIR,
                    help='PDF directory: The directory containing PDF files.')

# Define the device to use (CPU or GPU)
parser.add_argument('--gpu',
                    action='store_true',
                    help='Device: Use GPU instead of CPU (default). This is used to specify the device to use.')

args = parser.parse_args()
# Assign the values to the variables

if args.vstoreName is not DEFAULT_VSTORE_NAME:
    vstoreName = add_trailing_slash(args.vstoreName)
else:
   vstoreName = add_trailing_slash(DEFAULT_PATH+args.vstoreName)

if args.vstoreDir is not DEFAULT_VSTORE_DIR:
    vstoreDir = add_trailing_slash(args.vstoreDir)
else:
   vstore = add_trailing_slash(DEFAULT_PATH+args.vstoreDir)

if args.modelDir is not DEFAULT_MODEL_DIR:
    modelDir = add_trailing_slash(args.modelDir)
else:
   modelDir = add_trailing_slash(DEFAULT_PATH+args.modelDir)

if args.modelName is not DEFAULT_MODEL_NAME:
    modelName = add_trailing_slash(args.modelName)
else:
   modelName = add_trailing_slash(DEFAULT_PATH+args.modelName)

if args.pdfDir is not DEFAULT_PDF_DIR:
    pdfDir = add_trailing_slash(args.pdfDir)
else:
    pdfDir = add_trailing_slash(DEFAULT_PATH+args.pdfDir)
gpu = args.gpu
modelPath = modelDir+modelName
vstorePath=vstoreDir+vstoreName

if not gpu:
    # Set default device to CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_device('cpu')

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
        print_help_and_exit(parser)
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
                # Add documents to FAISS index
                vector_store.add_documents(documents=pages, ids=uuids)

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