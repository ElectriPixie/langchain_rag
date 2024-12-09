import os
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from colorama import Fore, Back, Style

vstoreDir = "faiss_store/"
vstoreName = 'Book_Collection'
embedded_model = "all-MiniLM-L6-v2"
embedded_model_kwargs = {"device": "cpu"}
user = "User: "
chatmodel = "\nLlama-RAG - "
bright = Style.BRIGHT 
dim = Style.DIM
reset = Style.RESET_ALL
usercolor = reset+bright+Fore.GREEN
usertext = reset+bright+Fore.CYAN
ragcolor = reset+bright+Fore.GREEN
ragtext = reset+bright+Fore.CYAN

embeddings = HuggingFaceEmbeddings(model_name=embedded_model, model_kwargs=embedded_model_kwargs)
vector_store = FAISS.load_local(
    vstoreDir+vstoreName, embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})


chat = ChatOpenAI(model_name="llama-2-chat",
                  openai_api_base="http://localhost:6589/v1",
                  openai_api_key="sk-xxx", 
                  max_tokens=2048,
                  temperature=0.7)

def chatFunc(message, retriever):
    knowledge_base = "Book Collection: \n" 
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
    if response.content == 'NA':
        #print("context_response: "+response.content)
        knowledge_base = "General Knowledge: \n"
        messages = [
            SystemMessage(
                content=f"""You are an ai assistant that answers questions"""
            ),
            HumanMessage(content=message),
        ]
        response = chat.invoke(messages)
    return response, knowledge_base

while True:
    message = input(usercolor + user + reset + "\n" + usertext)
    if message == "goodbye":
        print(reset)
        break
    response, knowledge_base = chatFunc(message, retriever)
    print(ragcolor + chatmodel + knowledge_base + ragtext + response.content+"\n"+reset)