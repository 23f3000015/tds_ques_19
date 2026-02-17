from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from openai import OpenAI

client = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    docs: List[str]
    query: str


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.post("/similarity")
def similarity(data: RequestBody):

    # Generate embeddings
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[data.query] + data.docs
    )

    embeddings = [item.embedding for item in response.data]

    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    scores = []

    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((score, data.docs[i]))

    scores.sort(reverse=True)

    top_3 = [doc for _, doc in scores[:3]]

    return {"matches": top_3}
