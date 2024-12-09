from langchain_openai import ChatOpenAI
import time
import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    model="Llama",
    temperature=0.7, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-xxx",
    base_url="http://localhost:6589/v1",
)

consoleLog = 1
chatuser = "User: \n"
chatmodel = "Llama-RAG - "
vstoreDir = "faiss_store/"
vstoreName = 'Book_Collection'
embedded_model = "all-MiniLM-L6-v2"
embedded_model_kwargs = {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(model_name=embedded_model, model_kwargs=embedded_model_kwargs)
vector_store = FAISS.load_local(
    vstoreDir+vstoreName, embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

def chatFunc(message, history):
    knowledge_base = "Book Collection: \n" 
    if consoleLog:
        print(chatuser+message+"\n")
    retrieved_documents = retriever.invoke(message)
    db = FAISS.from_documents(retrieved_documents, embeddings)
    docs = db.similarity_search(message)

    messages = [
        SystemMessage(
            content=f"""You are an ai assistant that answers questions searching for a response only the current context, citing book sources when available, if the question cannot be answered directly from the context then answer only with 'NA'

            Current context:
            {docs[0].page_content}
            """
        ),
        HumanMessage(content=message),
    ]

    response = chat.invoke(messages)
    #print("context_response: "+response.content)
    if response.content == 'NA':
        knowledge_base = "General Knowledge: \n"
        messages = [
            SystemMessage(
                content=f"""You are an ai assistant that answers questions"""
            ),
            HumanMessage(content=message),
        ]
        response = chat.invoke(messages)
    if consoleLog:
        print(chatmodel+knowledge_base+response.content)
    return chatmodel+knowledge_base+"\n"+response.content

gr.ChatInterface(chatFunc, type="messages").launch()