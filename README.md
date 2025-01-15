# langchain_rag


## openai compatible LLM langchain rag implementation 

I use this with a local llama.cpp running, it will require a compatible backend interface

run ```./convert_model.sh``` or ```python3 convert_model.py``` to save the model for reuse without using unsafe deserializations 

then run ```./langchain_initialize_faiss.sh``` or ```python3 langchain_initialize_faiss.py``` to initialize the faiss store

then run ```./langchain_load_pdfs.sh``` or ```python3 langchain_load_pdfs.py``` to load pdfs into the faiss store

then run either ```./langchain_openai_faiss.sh``` or ```python3 langcahin_openai_faiss.py``` for command line chat to the chatbot with RAG
or run ```./gradio_langchain_rag_server.sh``` or ```python3 gradio_langchain_rag_server.py``` to get a web interface to the chatbot with RAG