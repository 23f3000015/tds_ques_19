from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("AIPIPE_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

class RequestModel(BaseModel):
    docs: List[str]
    query: str

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/similarity")
def similarity(request: RequestModel):
    # Embed query
    query_embedding = get_embedding(request.query)

    # Embed all documents in ONE call
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.docs
    )

    doc_embeddings = [np.array(e.embedding) for e in response.data]

    scores = []
    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((i, score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    top_3 = [index for index, _ in ranked[:3]]

    return {"matches": top_3}
