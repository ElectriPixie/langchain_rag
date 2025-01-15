#!/bin/bash

REL_SCRIPT_DIR=$(dirname "$0")
source $REL_SCRIPT_DIR/sharedFunctions/sharedFunctions.sh
source $REL_SCRIPT_PATH/config/readConfig.sh
SCRIPT_DIR=$(get_script_dir "$0")
DEFAULT_PATH=$(dirname $SCRIPT_DIR)

SCRIPT_NAME="initializeFaissStore.py"
SCRIPT_PATH=$DEFAULT_PATH/pylib/$SCRIPT_NAME

# Set default values
vstoreName=$DEFAULT_VSTORE_NAME
vstoreDir=$DEFAULT_PATH/$DEFAULT_VSTORE_DIR
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