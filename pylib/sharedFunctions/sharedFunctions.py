import psutil
import os
import sys

def add_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path

def get_program_name():
    parent_pid = os.getppid()
    parent_process = psutil.Process(parent_pid)
    parent_cmdline = parent_process.cmdline()

    if len(parent_cmdline) > 1:  # Check if there are arguments
        run_script_name = os.path.basename(parent_cmdline[1])  # Get only the file name
    else:
        run_script_name = "Unknown"

    script_name = os.path.basename(__file__)

    run_script_base = os.path.splitext(run_script_name)[0]
    script_base = os.path.splitext(script_name)[0]

    if run_script_base == script_base:
        return run_script_name
    else:
        return script_name

def print_help_and_exit(parser):
    parser.print_help()
    sys.exit(0)