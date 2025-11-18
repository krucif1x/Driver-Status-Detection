#!/bin/bash
venv_dir="$(pwd)/venv"
echo "Starting drowsiness detection with virtual environment at $venv_dir"
source "$venv_dir/bin/activate" && python main.py