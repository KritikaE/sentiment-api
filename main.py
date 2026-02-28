from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini uses OpenAI-compatible API!
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class Comment(BaseModel):
    comment: str

class Sentiment(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment")
async def analyze(req: Comment):
    try:
        res = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "Analyze sentiment. sentiment: positive/negative/neutral, rating: 1-5."},
                {"role": "user", "content": req.comment}
            ],
            response_format=Sentiment
        )
        return res.choices[0].message.parsed
    except Exception as e:
        return {"error": str(e)}
