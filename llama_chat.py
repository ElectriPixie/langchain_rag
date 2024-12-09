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

user = "User: "
chatmodel = "\nLlama: "
bright = Style.BRIGHT 
dim = Style.DIM
reset = Style.RESET_ALL
usercolor = reset+bright+Fore.GREEN
usertext = reset+bright+Fore.CYAN
ragcolor = reset+bright+Fore.GREEN
ragtext = reset+bright+Fore.CYAN

chat = ChatOpenAI(model_name="llama-2-chat",
                  openai_api_base="http://localhost:6589/v1",
                  openai_api_key="sk-xxx", 
                  max_tokens=2048,
                  temperature=0.7)

def chatFunc(message):
    messages = [
        SystemMessage(
            content=f"""you are an ai assistance happy to help with any request"""
        ),
        HumanMessage(content=message),
    ]
    response = chat.invoke(messages)
    return response

while True:
    message = input(usercolor + user + reset + "\n" + usertext)
    if message == "goodbye":
        print(reset)
        break
    response = chatFunc(message)
    print(ragcolor + chatmodel + ragtext + response.content+"\n"+reset)