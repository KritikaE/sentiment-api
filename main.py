from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import json
import os

app = FastAPI()
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
            response_format={"type": "json_object"},  # forces JSON output
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response as JSON"}
    except Exception as e:
        return {"error": str(e)}
