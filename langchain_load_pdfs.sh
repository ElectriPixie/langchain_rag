#!/bin/bash

# Set default values
vstoreName="Book_Collection"
vstoreDir="faiss_store"
pdfDir="pdf"
modelPath="all-MiniLM-L6-v2/"
cpu="False"
perPageEmbeddings="False"
help="False"

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
    --pdfDir)
      if [[ -n $2 && $2 != -* ]]; then
        pdfDir="$2"
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
    --perPageEmbeddings)
      perPageEmbeddings="True"
      shift
      ;;
    --cpu)
      cpu="True"
      shift
      ;;
    --help)
      help="True"
      shift
      ;;
    -h)
      help="True"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Run the Python script with arguments
python3 langchain_load_pdfs.py \
  --vstoreName    "$vstoreName"  \
  --vstoreDir    "$vstoreDir"  \
  --pdfDir       "$pdfDir"     \
  --modelPath "$modelPath" \
  --perPageEmbeddings "$perPageEmbeddings" \
  $([ "$cpu" = "True" ] && echo "--cpu") \
  $([ "$help" = "True" ] && echo "--help")