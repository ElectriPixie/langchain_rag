#!/bin/bash

# Set default values
modelName="all-MiniLM-L6-v2"
cpu="False"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --modelName)
      if [[ -n $2 & $2 != -* ]]; then
        modelName="$2"
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
python3 convert_model.py \
  --model_name "$modelName" \
  --cpu "$cpu"