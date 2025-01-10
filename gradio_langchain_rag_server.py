import gradio as gr
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
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from colorama import Fore, Style
from langchain.schema import Document
import argparse

parser = argparse.ArgumentParser()

# Define the arguments
parser.add_argument('--vstoreName', type=str, default='Book_Collection')
parser.add_argument('--vstoreDir', type=str, default='faiss_store')
parser.add_argument('--model_load_path', type=str, default='all-MiniLM-L6-v2/')
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
vstorePath = vstoreDir+vstoreName
model_load_path = args.model_load_path
cpu = args.cpu

if cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define the custom embeddings class that inherits from LangChain's Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_load_path: str):
        self.model = SentenceTransformer(model_load_path)

    def embed_query(self, query: str):
        if cpu: 
            return self.model.encode([query], convert_to_tensor=True)[0].cpu().numpy()
        else:
            return self.model.encode([query], convert_to_tensor=True)[0].numpy()


    def embed_documents(self, documents: list):
        return [self.embed_query(doc) for doc in documents]

# Set default device to CPU
torch.set_default_device('cpu')
#torch.cuda.empty_cache()

# Paths and settings
chatuser = "User: \n"
chatmodel = "Llama-RAG - "


# Create custom embeddings object and load the saved model weights (embeddings.pt)
embeddings = SentenceTransformerEmbeddings(model_load_path=model_load_path)
if cpu:
    embeddings.model.to('cpu')

# Load documents from the JSON file
try:
    with open(os.path.join(vstorePath, "documents.json"), "r") as doc_file:
        docstore_data = json.load(doc_file)
except json.JSONDecodeError as e:
    print(f"Error loading documents.json: {e}")
    docstore_data = {}

# Recreate the FAISS index
index = faiss.IndexFlatL2(embeddings.model.get_sentence_embedding_dimension())

# Create the FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,  # Pass the custom embeddings object
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Add the documents to the vector store
for doc_id, doc_data in docstore_data.items():
    # Extract text and metadata from the stored document data
    doc_text = doc_data.get('text', '')
    doc_metadata = doc_data.get('metadata', {})
    
    # Create a Document object with text and metadata
    document = Document(
        page_content=doc_text,
        metadata=doc_metadata
    )

    # Add the document to the FAISS index and the InMemoryDocstore
    vector_store.add_documents([document], doc_ids=[doc_id])

    # The metadata is automatically handled by the docstore, so no need for manual assignment

# Load the FAISS index from file
vector_store.index = faiss.read_index(os.path.join(vstorePath, f"index.faiss"))

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

# Initialize the chat model
chat = ChatOpenAI(
    model_name="llama-2-chat",
    openai_api_base="http://localhost:6589/v1",
    openai_api_key="sk-xxx",
    max_tokens=2048,
    temperature=0.7
)

# Define the function to interact with the chat model
def chatFunc(message):
    knowledge_base = "Book Collection: \n"
    # Search for relevant documents based on the message
    retrieved_documents = retriever.invoke(message)
    
    #if not retrieved_documents:
    #    knowledge_base = "General Knowledge: \n"
    #    return chat_with_general_knowledge(message)
    
    # Create a retriever from the documents and use similarity search
    db = FAISS.from_documents(retrieved_documents, embeddings)
    docs = db.similarity_search(message)
    
    # Add context to the conversation
    messages = [
        SystemMessage(
            content=f"""You are an AI assistant that answers questions searching for a response only from the current context, citing book sources when available. If the question cannot be answered directly from the context, then answer only with 'NA'.

            Current context:
            {docs[0].page_content}
            """
        ),
        HumanMessage(content=message),
    ]
    
    # Get response from the chat model
    response = chat.invoke(messages)
    if response.content == 'NA' or response.content == "Not found in current context.":
        #print("context_response: "+response.content)
        knowledge_base = "General Knowledge: \n"
        messages = [
            SystemMessage(
                content=f"""You are an ai assistant that answers questions"""
            ),
            HumanMessage(content=message),
        ]
        response = chat.invoke(messages)
    return chatmodel+knowledge_base+"\n"+response.content

# Fallback to general knowledge when no relevant documents are found
def chat_with_general_knowledge(message):
    messages = [
        SystemMessage(
            content="You are an AI assistant that answers questions with general knowledge."
        ),
        HumanMessage(content=message),
    ]
    response = chat.invoke(messages)
    return response.content

# Wrapper function to pass the `retriever` to `chatFunc`
def wrapped_chatFunc(message, type):
    return chatFunc(message)

# Use the wrapper in the Gradio interface
gr.ChatInterface(wrapped_chatFunc, type="messages").launch()