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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define the custom embeddings class that inherits from LangChain's Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_load_path: str):
        self.model = SentenceTransformer(model_load_path)

    def embed_query(self, query: str):
        return self.model.encode([query], convert_to_tensor=True)[0].cpu().numpy()

    def embed_documents(self, documents: list):
        return [self.embed_query(doc) for doc in documents]

# Paths and settings
vstoreName = "Book_Collection"
vstoreDir = "faiss_store/" + vstoreName + "/"
pdfDir = "/home/pixie/AI/ai_tools/Llama/data/pdf/"
model_load_path = 'all-MiniLM-L6-v2/'

# Set default device to CPU
torch.set_default_device('cpu')

# Create custom embeddings object
embeddings = SentenceTransformerEmbeddings(model_load_path=model_load_path)

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
    """
    Converts a Document object to a dictionary for serialization.
    
    Args:
    - doc: The Document object to convert.
    
    Returns:
    - dict: A dictionary representing the Document object.
    """
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

# Function to add documents to the FAISS index and the document store
def add_documents_to_store(pdfDir):
    all_documents = []  # List to store documents with their UUIDs
    for file in os.listdir(pdfDir):
        if file.endswith('.pdf'):
            pdfName = file
            print("Loading: " + pdfName)

            file_path = os.path.join(pdfDir, pdfName)
            loader = PyPDFLoader(file_path)
            pages = []
            for page in loader.lazy_load():
                pages.append(page)

            # Create UUIDs for the documents
            uuids = [str(uuid4()) for _ in range(len(pages))]
            
            # Add documents to FAISS index
            vector_store.add_documents(documents=pages, ids=uuids)

            # Store documents and their UUIDs for saving
            for uuid, page in zip(uuids, pages):
                all_documents.append({"uuid": uuid, "content": page})

    return all_documents

# Add documents to the vector store and retrieve document content
all_documents = add_documents_to_store(pdfDir)

# Save the FAISS index directly
faiss.write_index(index, os.path.join(vstoreDir, f"index.faiss"))

# Save the document store (in memory)
docstore_data = {doc["uuid"]: document_to_dict(doc["content"]) for doc in all_documents}

with open(os.path.join(vstoreDir, f"documents.json"), "w") as doc_file:
    json.dump(docstore_data, doc_file)

print(f"Documents and FAISS index saved to {vstoreDir}")