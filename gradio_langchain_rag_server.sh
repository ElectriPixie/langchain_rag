#!/bin/bash

# Set default values
vstoreName="Book_Collection"
vstoreDir="faiss_store"
modelPath="all-MiniLM-L6-v2/"
cpu="False"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --vstoreName)
      if [[ -n $2 && $2 != -* ]]; then
        vstoreName="$2"
        shift 2
      else
        shift # Skip invalid value, keep default
      fi
      ;;
    --vstoreDir)
      if [[ -n $2 && $2 != -* ]]; then
        vstoreDir="$2"
        shift 2
      else
        shift # Skip invalid value, keep default
      fi
      ;;
    --modelPath)
      if [[ -n $2 && $2 != -* ]]; then
        modelPath="$2"
        shift 2
      else
        shift # Skip invalid value, keep default
      fi
      ;;
    --cpu)
      cpu="True"
      shift
      ;;
  esac
done

# Run the Python script with arguments
python3 gradio_langchain_rag_server.py \
  --vstoreName    "$vstoreName" \
  --vstoreDir    "$vstoreDir" \
  --modelPath   "$modelPath" \
  --cpu          "$cpu"