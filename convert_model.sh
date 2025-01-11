#!/bin/bash

# Set default values
modelName="all-MiniLM-L6-v2"
cpu="False"
help="False"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --modelName)
      if [[ -n $2 ]]; then
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
python3 convert_model.py \
  --modelName "$modelName" \
  $([ "$cpu" = "True" ] && echo "--cpu") \
  $([ "$help" = "True" ] && echo "--help")