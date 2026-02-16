from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

def text_to_vector(text):
    return np.array([ord(c) for c in text])

def cosine_similarity(a, b):
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/similarity")
def similarity(request: RequestModel):
    query_vec = text_to_vector(request.query)

    scores = []
    for doc in request.docs:
        doc_vec = text_to_vector(doc)
        score = cosine_similarity(query_vec, doc_vec)
        scores.append(score)

    ranked = sorted(
        zip(request.docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_3 = [doc for doc, _ in ranked[:3]]

    return {"matches": top_3}
