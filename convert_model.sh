#!/bin/bash

# Set default values
model_name="all-MiniLM-L6-v2"
cpu="False"

# Parse command-line arguments
for arg in "$@"; do
  case $arg in
    --model-name*)
      model_name="${arg#--model-name=}"
      ;;
    --cpu*)
      cpu=true
      ;;
  esac
done

# Run the Python script with arguments
python3 convert_model.py --model_name $model_name --cpu $cpu