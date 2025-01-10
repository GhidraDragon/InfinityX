#!/usr/bin/env bash

# 1. Check if python3 is installed.
#    If python3 is not installed, you will need to install it before proceeding.
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found, please install Python 3 and try again."
    exit 1
fi

# 2. Create a virtual environment in a folder named '.venv' (change the name if you like).
echo "Creating a virtual environment in the '.venv' directory..."
python3 -m venv .venv

# 3. Check if the .venv folder was created successfully.
if [ ! -d ".venv" ]; then
  echo "Failed to create virtual environment."
  exit 1
fi

# 4. Activate the newly created environment.
echo "Activating the virtual environment..."
source .venv/bin/activate

# 5. Install an example package (optional step).
echo "Installing an example package: requests"
pip install requests

# 6. Let the user know the environment is ready.
echo "Virtual environment is ready and activated. You can now run Python with the environment."
echo "To deactivate the environment, type: deactivate"
