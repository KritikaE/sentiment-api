from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import json
import os

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system="""You are a sentiment analysis assistant.
Analyze the sentiment of the user's comment.
You MUST respond with ONLY valid JSON in this exact format, nothing else:
{"sentiment": "positive", "rating": 5}

Rules:
- sentiment must be exactly one of: "positive", "negative", "neutral"
- rating must be an integer from 1 to 5 (5=very positive, 1=very negative)
- No explanation, no markdown, no code blocks, just the raw JSON""",
            messages=[
                {"role": "user", "content": request.comment}
            ]
        )

        result = json.loads(response.content[0].text.strip())
        return result

    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response as JSON"}
    except Exception as e:
        return {"error": str(e)}
