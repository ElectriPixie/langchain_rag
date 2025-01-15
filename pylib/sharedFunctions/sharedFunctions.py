import psutil
import os
import sys

def add_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path

def print_help_and_exit(parser):
    parser.print_help()
    sys.exit(0)