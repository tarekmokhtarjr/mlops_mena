FROM astral/uv:python3.12-bookworm-slim

SHELL ["/bin/bash", "-c"]

# Create the necessary directories
RUN mkdir -p /minilm
RUN mkdir -p /minilm/models

# Install huggingface_hub
RUN uv tool install huggingface_hub

# Download the model all-MiniLM-L6-v2
RUN hf download \
    sentence-transformers/all-MiniLM-L6-v2 \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    vocab.txt \
    --local-dir /minilm/models/all-MiniLM-L6-v2

# Copy pyproject.toml and app files to the correct location
COPY pyproject.toml /minilm/pyproject.toml
COPY README.md /minilm/README.md
COPY ./src minilm/src

# Set the working directory
WORKDIR /minilm/src/app

# Create and activate a virtual environment using uv tool
RUN uv venv --python 3.12
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies from pyproject.toml
RUN uv sync

ENTRYPOINT ["uvicorn", "main:app"]
