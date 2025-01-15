import config
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
import config

parser = argparse.ArgumentParser()
parser.add_argument('--DEFAULT_MODEL_DIR', action='store_true')
parser.add_argument('--DEFAULT_MODEL_NAME', action='store_true')
parser.add_argument('--DEFAULT_VSTORE_DIR', action='store_true')
parser.add_argument('--DEFAULT_VSTORE_NAME', action='store_true')
parser.add_argument('--DEFAULT_PDF_DIR', action='store_true')

args = parser.parse_args()

if args.DEFAULT_MODEL_DIR:
    print(getattr(config, 'DEFAULT_MODEL_DIR'))
if args.DEFAULT_MODEL_NAME:
    print(getattr(config, 'DEFAULT_MODEL_NAME'))
if args.DEFAULT_VSTORE_DIR:
    print(getattr(config, 'DEFAULT_VSTORE_DIR'))
if args.DEFAULT_VSTORE_NAME:
    print(getattr(config, 'DEFAULT_VSTORE_NAME'))
if args.DEFAULT_PDF_DIR:
    print(getattr(config, 'DEFAULT_PDF_DIR'))