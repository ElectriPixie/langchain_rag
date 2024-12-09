import os

from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader

pdfDir = "/home/pixie/AI/ai_tools/Llama/data/pdf/"
vstoreDir = "faiss_store/"
vstoreName = "Book_Collection"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

vector_store = FAISS.load_local(
    vstoreDir+vstoreName, embeddings, allow_dangerous_deserialization=True
)

for file in os.listdir(pdfDir):
    if file.endswith('.pdf'):
        pdfName = file
        print("Loading: " + pdfName)

    file_path = (
        pdfDir+pdfName
    )
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    uuids = [str(uuid4()) for _ in range(len(pages))]
    vector_store.add_documents(documents=pages, ids=uuids)

vector_store.save_local(vstoreDir+vstoreName)