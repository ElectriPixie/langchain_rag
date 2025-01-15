# langchain_rag


## openai compatible LLM langchain rag implementation 

I use this with a local llama.cpp running, it will require a compatible backend interface

run ```./sbin/convertModel.sh``` or ```python3 pylib/convertModel.py``` to save the model for reuse without using unsafe deserializations 

then run ```./sbin/initializeFaissStore.sh``` or ```python3 pylib/initializeFaissStore.py``` to initialize the faiss store

add pdfs to pdf directory the default is ```pdf/``` in the project root create with ```mkdir pdf```

then run ```./sbin/loadPdfs.sh``` or ```python3 pylib/convertModel.py``` to load pdfs into the faiss store it will skip pdfs it's already added so more can be added and it can be rerun to update the faiss store

then run either ```./sbin/chatRag.sh``` or ```python3 pylib/chatRag.py``` for command line chat or ```./sbin/gradioRagServer.sh``` or ```python3 pylib/gradioRagServer.py``` to get a web interface to the LLM backed chatbot with RAG services

default values are configured in ```config/config.py```

all the scripts have help strings so you can use ```--help``` or ```-h``` to get an explaination of options

the bash scripts should run on most posix compliant operating systems. Hopefully it works for everyone but it's just a wrapper you can call the scripts directly