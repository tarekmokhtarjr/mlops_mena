#!/bin/bash

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install venv using uv
echo "Creating virtual environment..."
uv venv --python 3.12

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if huggingface hf tool is installed, if not install it
if ! command -v hf &> /dev/null; then
    echo "huggingface hf tool is not installed. Installing..."
    uv tool install huggingface_hub
fi

# Create folder models
mkdir -p models

# Check Dockerfile and download all-MiniLM-L6-v2 using hf
echo "Downloading all-MiniLM-L6-v2 using hf..."
hf download \
    sentence-transformers/all-MiniLM-L6-v2 \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    vocab.txt \
    --local-dir models/all-MiniLM-L6-v2

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
uv sync

# Install dev group dependencies in pyproject.toml
echo "Installing dev dependencies..."
uv sync --groupdev

# Install pre-commit hooks using uv run pre-commit install
echo "Installing pre-commit hooks..."
uv run pre-commit install

echo "Setup complete."
