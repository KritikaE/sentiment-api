from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os
import asyncio
import httpx

app = FastAPI()

# Fix CORS so the grader can reach your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a sentiment analysis assistant.
Respond with ONLY valid JSON, nothing else. No markdown, no explanation.
Format: {"sentiment": "positive", "rating": 5}
- sentiment: exactly "positive", "negative", or "neutral"
- rating: integer 1-5 (5=very positive, 1=very negative)"""
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response as JSON"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"status": "alive"}

async def keep_alive():
    await asyncio.sleep(60)
    async with httpx.AsyncClient() as http_client:
        while True:
            try:
                await http_client.get("https://sentiment-api-7acd.onrender.com/")
            except:
                pass
            await asyncio.sleep(600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive())
