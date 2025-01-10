#!/bin/bash

# Set default values
vstoreName="Book_Collection"
vstoreDir="faiss_store"
model_load_path="all-MiniLM-L6-v2/"
cpu="False"

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --vstoreName*)
      vstoreName="${arg#--vstoreName=}"
      ;;
    --vstoreDir*)
      vstoreDir="${arg#--vstoreDir=}"
      ;;
    --model_load_path*)
      model_load_path="${arg#--model_load_path=}"
      ;;
    --cpu*)
      cpu="True"
      ;;
  esac
done

# Run the Python script with arguments
python3 langchain_openai_faiss.py --vstoreName $vstoreName --vstoreDir $vstoreDir --model_load_path $model_load_path --cpu $cpu