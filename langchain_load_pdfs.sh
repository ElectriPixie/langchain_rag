#!/bin/bash

# Set default values
vstoreName="Book_Collection"
vstoreDir="faiss_store"
pdfDir="pdf"
model_load_path="all-MiniLM-L6-v2/"
cpu="False"
perPageEmbeddings="False"

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --vstoreName*)
      vstoreName="${arg#--vstoreName=}"
      ;;
    --vstoreDir*)
      vstoreDir="${arg#--vstoreDir=}"
      ;;
    --pdfDir*)
      pdfDir="${arg#--pdfDir=}"
      ;;
    --model_load_path*)
      model_load_path="${arg#--model_load_path=}"
      ;;
    --perPageEmbeddings*)
      perPageEmbeddings="True"
      ;;
    --cpu*)
      cpu="True"
      ;;
  esac
done

# Run the Python script with arguments
python3 langchain_load_pdfs.py --vstoreName $vstoreName --vstoreDir $vstoreDir --pdfDir $pdfDir --model_load_path $model_load_path --perPageEmbeddings $perPageEmbeddings --cpu $cpu