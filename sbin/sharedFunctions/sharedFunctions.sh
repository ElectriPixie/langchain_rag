#!/usr/bin/bash

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