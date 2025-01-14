# langchain_rag


## openai compatible LLM langchain rag implementation 

I use this with a local llama.cpp running, it will require a compatible backend interface

run ```./sbin/convertModel.py``` to save the model for reuse without using unsafe deserializations 

then run ```./sbin/initializeFaissStore.sh``` to initialize the faiss store

add pdfs to pdf directory the default is ```pdf/``` in the project root create with ```mkdir pdf```

then run ```./sbin/loadPdfs.sh``` to load pdfs into the faiss store it will skip pdfs it's already added so more can be added and it can be rerun to update the faiss store

then run either ```./sbin/chatRag.sh``` for command line chat or ```./sbin/gradioRagServer.sh``` to get a web interface to the LLM backed chatbot with RAG services