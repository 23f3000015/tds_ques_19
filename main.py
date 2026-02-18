from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from openai import OpenAI

app = FastAPI()

# ✅ CORS configuration (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route (helps prevent fetch errors)
@app.get("/")
def root():
    return {"status": "running"}

# ✅ Handle preflight explicitly (extra safety)
@app.options("/similarity")
def options_similarity():
    return {}

# ✅ AI Pipe Client
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://api.aipipe.org/v1"
)

class RequestBody(BaseModel):
    docs: List[str]
    query: str

@app.post("/similarity")
def similarity(data: RequestBody):

    # Generate embeddings
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[data.query] + data.docs
    )

    embeddings = [item.embedding for item in response.data]

    query_embedding = np.array(embeddings[0])
    doc_embeddings = [np.array(e) for e in embeddings[1:]]

    similarities = []

    for i, doc_embedding in enumerate(doc_embeddings):
        score = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((score, data.docs[i]))

    # Sort descending
    similarities.sort(key=lambda x: x[0], reverse=True)

    top_3 = [doc for _, doc in similarities[:3]]

    return {"matches": top_3}
