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
    onnx/model.onnx \
    --local-dir /minilm/models/all-MiniLM-L6-v2

# Copy pyproject.toml and app files to the correct location
COPY pyproject.toml /minilm/pyproject.toml
COPY ./src minilm/src

# Set the working directory
WORKDIR /minilm

# Create and activate a virtual environment using uv tool
RUN uv venv --python 3.12
ENV PATH="/minilm/.venv/bin:$PATH"

# Install dependencies from pyproject.toml
RUN uv sync

CMD ["uvicorn", "--app-dir", "src/app", "main:app", "--host", "0.0.0.0", "--port", "8000"]
