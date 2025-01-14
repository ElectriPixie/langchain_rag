#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f $BASH_SOURCE[0]))
DEFAULT_PATH=$(dirname $SCRIPT_DIR)
SCRIPT_NAME="convertModel.py"
SCRIPT_PATH=$DEFAULT_PATH/pylib/$SCRIPT_NAME

#source default values shared by python scripts
source $DEFAULT_PATH/pylib/config/config.py

# Set default values
modelName=$DEFAULT_MODEL_NAME
modelDir=$DEFAULT_PATH/$DEFAULT_MODEL_DIR
gpu="False"
help="False"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --modelName)
      if [[ -n $2 && $2 != -* ]]; then
        modelName="$2"
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
  --modelName "$modelName" \
  --modelDir "$modelDir" \
  $([ "$gpu" = "True" ] && echo "--gpu") \
  $([ "$help" = "True" ] && echo "--help")