#!/bin/bash

#path to the virtual environment
VENV_PATH="/home/adenzler/.venvs/efrey-mjx"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run your desired command
"$@"

# Deactivate the virtual environment (optional)
deactivate
