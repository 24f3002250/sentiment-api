from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Batch Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()

class SentimentRequest(BaseModel):
    sentences: list[str] = Field(min_length=1)

def analyze_sentiment(sentence: str) -> str:
    score = analyzer.polarity_scores(sentence)["compound"]

    if score >= 0.05:
        return "happy"
    elif score <= -0.05:
        return "sad"
    return "neutral"

@app.get("/")
def root():
    return {"message": "Batch Sentiment Analysis API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/")
@app.post("/sentiment")
def batch_sentiment(request: SentimentRequest):
    return {
        "results": [
            {
                "sentence": sentence,
                "sentiment": analyze_sentiment(sentence)
            }
            for sentence in request.sentences
        ]
    }
