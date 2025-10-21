from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Union, Dict
import time

app = FastAPI(title="MiniLM Embedding API")

model_cache: Dict[str, SentenceTransformer] = {}

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

@app.post("/embed")
async def embed(req: EmbeddingRequest):
    start = time.time()
    if req.model not in model_cache:
        model_cache[req.model] = SentenceTransformer(req.model)
    model = model_cache[req.model]

    if isinstance(req.text, list):
        embeddings = model.encode(req.text, convert_to_numpy=True).tolist()
    else:
        embeddings = model.encode([req.text], convert_to_numpy=True).tolist()[0]

    end = time.time()
    return {
        "model": req.model,
        "input_count": len(req.text) if isinstance(req.text, list) else 1,
        "processing_time": round(end - start, 3),
        "embeddings": embeddings
    }

@app.get("/")
def root():
    return {"status": "MiniLM API running 🚀"}
