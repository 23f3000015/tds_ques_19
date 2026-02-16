from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestModel(BaseModel):
    docs: List[str]
    query: str

def fake_embedding(text: str):
    return np.array([ord(c) % 50 for c in text[:50]])

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/similarity")
def similarity(request: RequestModel):
    query_emb = fake_embedding(request.query)

    scores = []
    for doc in request.docs:
        doc_emb = fake_embedding(doc)
        score = cosine_similarity(query_emb, doc_emb)
        scores.append(score)

    ranked_docs = sorted(
        zip(request.docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_3 = [doc for doc, _ in ranked_docs[:3]]

    return {"matches": top_3}
