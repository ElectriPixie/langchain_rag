# Read config file and export variables
read_config_file() {
  while read line; do
    if [[ $line =~ ^[A-Z_]+[[:space:]]*=[[:space:]]*[^#]+ ]]; then
      var=${line%%=*}
      value=${line#*=}
      value=$(echo "$value" | sed "s/['\"]//g")
      if [[ $var =~ ^DEFAULT_[A-Z_]+$ ]]; then
        export $var=$value
      else
        echo "Warning: skipping variable $var"
      fi
    fi
  done < "$1"
}