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

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Comment(BaseModel):
    comment: str

class Sentiment(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment")
async def analyze(req: Comment):
    try:
        res = client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Analyze sentiment. sentiment: positive/negative/neutral, rating: 1-5."},
                {"role": "user", "content": req.comment}
            ],
            response_format=Sentiment
        )
        return res.choices[0].message.parsed
    except Exception as e:
        return {"error": str(e)}
