from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

class RequestBody(BaseModel):
    docs: List[str]
    query: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def similarity(data: RequestBody):

    embeddings = model.encode([data.query] + data.docs)

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    scores = []

    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((score, data.docs[i]))

    scores.sort(reverse=True)

    top_3 = [doc for _, doc in scores[:3]]

    return {"matches": top_3}
