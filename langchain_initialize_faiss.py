import os
import json
import torch
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

pt_model = 'all-MiniLM-L6-v2.pt'
# Define the custom embeddings class that inherits from LangChain's Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model.load_state_dict(torch.load(pt_model, map_location=torch.device('cpu'), weights_only=True))

    def embed_query(self, query: str):
        return self.model.encode([query], convert_to_tensor=True)[0].cpu().numpy()

    def embed_documents(self, documents: list):
        return [self.embed_query(doc) for doc in documents]

# Paths and settings
vstoreName = "Book_Collection"
vstoreDir = "faiss_store/"+vstoreName+"/"
os.makedirs(vstoreDir, exist_ok=True)
embedded_model = "sentence-transformers/all-MiniLM-L6-v2"

# Set default device to CPU
torch.set_default_device('cpu')

# Create custom embeddings object
embeddings = SentenceTransformerEmbeddings(model_name=embedded_model)

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
faiss.write_index(index, os.path.join(vstoreDir, f"index.faiss"))

# Save the embeddings model weights
torch.save(embeddings.model.state_dict(), os.path.join(vstoreDir, f"embeddings.pt"))

# Create and save an empty document store (No documents added yet)
with open(os.path.join(vstoreDir, f"documents.json"), "w") as doc_file:
    json.dump({}, doc_file)

print(f"FAISS index and embeddings initialized and saved to {vstoreDir}. No documents added.")
