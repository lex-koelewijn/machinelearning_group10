#!/bin/bash

# This converts the requested .py to .ipynb
# Start the jupyter server
# Converts .ipynb back to .py once the server is closed

echo "[script] Enter filename you want to work on (without .py or .ipynb):"
read filename

echo "[script] Converting $filename.py to $filename.ipynb"
jupytext --to ipynb $filename.py
echo "[script] Starting Jupyter Lab server"
jupyter lab
echo "[script] Converting $filename.ipynb back to $filename.py"
jupytext --to py $filename.ipynb
echo "[script] Removing $filename.ipynb"
rm $filename.ipynb
echo "[script] Complete"
