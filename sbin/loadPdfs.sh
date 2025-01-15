#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f $BASH_SOURCE[0]))
DEFAULT_PATH=$(dirname $SCRIPT_DIR)
SCRIPT_NAME="loadPdfs.py"
SCRIPT_PATH=$DEFAULT_PATH/pylib/$SCRIPT_NAME

source $DEFAULT_PATH/sbin/config/readConfig.sh

# Set default values
vstoreName=$DEFAULT_VSTORE_NAME
vstoreDir=$DEFAULT_PATH/$DEFAULT_VSTORE_DIR
pdfDir=$DEFAULT_PATH/$DEFAULT_PDF_DIR
modelName=$DEFAULT_MODEL_NAME
modelDir=$DEFAULT_PATH/$DEFAULT_MODEL_DIR
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
    --pdfDir)
      if [[ -n $2 && $2 != -* ]]; then
        pdfDir="$2"
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
  --vstoreName    "$vstoreName"  \
  --vstoreDir    "$vstoreDir"  \
  --pdfDir       "$pdfDir"     \
  --modelDir   "$modelDir" \
  --modelName   "$modelName" \
  $([ "$gpu" = "True" ] && echo "--gpu") \
  $([ "$help" = "True" ] && echo "--help")