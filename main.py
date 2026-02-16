from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/similarity")
def similarity(req: SimilarityRequest):
    query_embedding = get_embedding(req.query)

    doc_embeddings = [get_embedding(doc) for doc in req.docs]

    scores = []
    for doc, emb in zip(req.docs, doc_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    top_matches = [doc for doc, _ in scores[:3]]

    return {
        "matches": top_matches
    }

