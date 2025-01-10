# langchain_rag


openai compatible LLM langchain rag implementation 

I use this with a local llama.cpp running, it will require a compatible interface

for release 1.0.5 and later
run ./convert_model.py to save the model for reuse without using unsafe deserializations 
then run ./langchain_initialize_faiss.sh to initialize the faiss store
then run ./langchain_load_pdfs.sh to load pdfs into the faiss store
then run either ./langchain_openai_faiss.sh or ./gradio_langchain_rag_server.sh to get an interface to the chatbot with RAG