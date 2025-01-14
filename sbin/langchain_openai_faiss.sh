#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f $BASH_SOURCE[0]))
DEFAULT_PATH=$(dirname $SCRIPT_DIR)
SCRIPT_NAME="langchain_openai_faiss.py"
SCRIPT_PATH=$DEFAULT_PATH/pylib/$SCRIPT_NAME

# Set default values
vstoreName="Book_Collection"
vstoreDir=$DEFAULT_PATH/faiss_store
modelName="all-MiniLM-L6-v2"
modelDir=$DEFAULT_PATH/models
gpu="False"
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
    --modelDir)
      if [[ -n $2 && $2 != -* ]]; then
        modelDir="$2"
        shift 2
      else
        shift # Skip invalid value, keep default
      fi
      ;;
    --modelName)
      if [[ -n $2 && $2 != -* ]]; then
        modelName="$2"
        shift 2
      else
        shift # Skip invalid value, keep default
      fi
      ;;
    --gpu)
      gpu="True"
      shift
      ;;
    -h|--help)
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
python3 ${SCRIPT_PATH} \
  --vstoreName    "$vstoreName" \
  --vstoreDir    "$vstoreDir" \
  --modelDir   "$modelDir" \
  --modelName   "$modelName" \
  $([ "$gpu" = "True" ] && echo "--gpu") \
  $([ "$help" = "True" ] && echo "--help")