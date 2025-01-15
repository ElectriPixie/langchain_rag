#!/bin/bash
python="python3"

get_script_dir() {
    local target_file="$1"
    
    # Function to resolve symlinks
    resolve_symlink() {
        local file="$1"
        while [ -L "$file" ]; do
            local dir
            dir="$(cd "$(dirname "$file")" && pwd -P)"
            file="$(ls -l "$file" | awk '{print $NF}')"
            [ "${file:0:1}" != "/" ] && file="$dir/$file"
        done
        echo "$file"
    }
    
    # Resolve the full path of the script
    target_file="$(resolve_symlink "$target_file")"
    
    # Return the absolute directory
    cd "$(dirname "$target_file")" && pwd -P
}

SCRIPT_DIR=$(get_script_dir "$0")
DEFAULT_PATH=$(dirname $SCRIPT_DIR)
source $DEFAULT_PATH/sbin/config/readConfig.sh

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
$python ${SCRIPT_PATH} \
  --vstoreName    "$vstoreName"  \
  --vstoreDir    "$vstoreDir"  \
  --pdfDir       "$pdfDir"     \
  --modelDir   "$modelDir" \
  --modelName   "$modelName" \
  $([ "$gpu" = "True" ] && echo "--gpu") \
  $([ "$help" = "True" ] && echo "--help")