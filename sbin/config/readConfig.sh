#!/bin/bash

# Read config file and export variables
#
# This function reads a config file and exports variables defined in it.
# It only exports variables that start with 'DEFAULT_'.
#
# Parameters:
#   $1: The path to the config file to read.
#
read_config_file() {
  # Read the config file line by line.
  while read line; do
    # Check if the line defines a variable.
    if [[ $line =~ ^[A-Z_]+[[:space:]]*=[[:space:]]*[^#]+ ]]; then
      # Extract the variable name and value from the line.
      var=${line%%=*}
      value=${line#*=}
      value=$(echo "$value" | sed "s/['\"]//g")
      
      # Check if the variable name starts with 'DEFAULT_'.
      if [[ $var =~ ^DEFAULT_[A-Z_]+$ ]]; then
        # Export the variable.
        export $var=$value
      else
        # If the variable name does not start with 'DEFAULT_', print a warning.
        echo "Warning: skipping variable $var"
      fi
    fi
  done < "$1"
}