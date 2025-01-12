# langchain_rag


## openai compatible LLM langchain rag implementation 

I use this with a local llama.cpp running, it will require a compatible backend interface

run ```./sbin/convert_model.py``` to save the model for reuse without using unsafe deserializations 

then run ```./sbin/langchain_initialize_faiss.sh``` to initialize the faiss store

add pdfs to pdf directory the default is ```pdf/``` in the project root

then run ```./sbin/langchain_load_pdfs.sh``` to load pdfs into the faiss store

then run either ```./sbin/langchain_openai_faiss.sh``` or ```./sbin/gradio_langchain_rag_server.sh``` to get an interface to the chatbot with RAG