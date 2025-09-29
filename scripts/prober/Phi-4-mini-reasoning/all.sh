#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

bash "$SCRIPT_DIR/prepare.sh"
bash "$SCRIPT_DIR/inference.sh"
bash "$SCRIPT_DIR/train.sh"
bash "$SCRIPT_DIR/test.sh"

