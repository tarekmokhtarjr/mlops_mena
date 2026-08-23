from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer
import os

app = FastAPI()
MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "models",
    "all-MiniLM-L6-v2",
)


class TextRequest(BaseModel):
    text: str


@app.get("/info")
def get_model_info() -> dict:
    """
    Returns information about the used model MiniLM-L6-v2.
    Returns:
        dict: A dictionary containing information about the used model.
    """
    description = """This is the ONNX-ported version of the sentence-transformers/all-MiniLM-L6-v2 for generating text embeddings.

Model details:
- Embedding dimension: 384
- Max sequence length: 256
- File size on disk: 0.08 GB
- Modules incorporated in the onnx: Transformer, Pooling, Normalize

"""
    return {
        "model": "MiniLM-L6-v2",
        "description": description,
        "repository": "https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx",
        "version": "v2",
        "license": "Apache-2.0",
        "architecture": "Transformer-based",
    }


@app.get("/embed")
async def get_embedding(
    text: str = Query(..., description="Text to generate embedding for")
) -> dict:
    """
    Generates an embedding for the given text.

    Args:
        text (str): The text to generate an embedding for.

    Returns:
        dict: A dictionary containing the tokens, token IDs, attention mask, and embedding.
    """
    if not text:
        return {"error": "Text parameter is required"}

    try:
        tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Tokenizer error: {str(e)}"
        )
    try:
        session = InferenceSession(
            f"{MODEL_DIR}/onnx/model.onnx", providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference session error: {str(e)}"
        )
    try:
        encoded = tokenizer.encode(text, add_special_tokens=True)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Encoding error: {str(e)}"
        )
    # Ensure the sequence is truncated to the maximum length
    max_length = 256
    input_ids = encoded.ids[:max_length]
    attention_mask = encoded.attention_mask[:max_length]
    token_type_ids = encoded.type_ids[:max_length]
    # Pad the sequence with zeros
    input_ids.extend([0] * (max_length - len(input_ids)))
    attention_mask.extend([0] * (max_length - len(attention_mask)))
    token_type_ids.extend([0] * (max_length - len(token_type_ids)))
    input_ids = np.array([input_ids])
    attention_mask = np.array([attention_mask])
    token_type_ids = np.array([token_type_ids])
    try:
        embedding = session.run(
            None,
            {
                session.get_inputs()[0].name: input_ids,
                session.get_inputs()[1].name: attention_mask,
                session.get_inputs()[2].name: token_type_ids,
            },
        )[0]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Session run error: {str(e)}"
        )
    return {
        "tokens": encoded.tokens[:max_length],
        "token_ids": input_ids.tolist()[0],
        "attention_mask": attention_mask.tolist()[0],
        "token_type_ids": token_type_ids.tolist()[0],
        "embedding": embedding.tolist(),
    }
