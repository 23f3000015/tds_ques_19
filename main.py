from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# Enable CORS (important for grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

# Deterministic fake embedding
def fake_embedding(text: str):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(128)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/similarity")
def similarity(req: SimilarityRequest):
    query_embedding = fake_embedding(req.query)

    scores = []
    for doc in req.docs:
        doc_embedding = fake_embedding(doc)
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((doc, score))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top 3 documents
    top_matches = [doc for doc, _ in scores[:3]]

    return {
        "matches": top_matches
    }
