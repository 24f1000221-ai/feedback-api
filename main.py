import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import JSONResponse

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Request schema
class CommentRequest(BaseModel):
    comment: str

# Response schema
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            raise HTTPException(status_code=400, detail="Comment cannot be empty")

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze sentiment of this comment: {request.comment}",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = response.output_parsed

        return JSONResponse(
            content=result,
            media_type="application/json"
        )

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))