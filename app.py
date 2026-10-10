from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Batch Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    sentences: list[str] = Field(min_length=1)

POSITIVE = {
    "love", "great", "excellent", "amazing", "awesome",
    "happy", "wonderful", "fantastic", "good", "best",
    "enjoy", "enjoyed", "like", "perfect", "beautiful",
    "delighted", "brilliant", "pleased", "thankful"
}

NEGATIVE = {
    "hate", "terrible", "awful", "sad", "bad", "worst",
    "horrible", "disappointed", "angry", "poor", "annoying",
    "boring", "ugly", "fail", "failed", "broken", "upset",
    "unhappy", "dislike"
}

NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't", "can't"}

def analyze_sentiment(sentence: str) -> str:
    words = sentence.lower().replace("n't", " not").split()
    positive_score = 0
    negative_score = 0

    for i, word in enumerate(words):
        word = word.strip(".,!?;:\"'()[]{}")
        if word in POSITIVE:
            score = 1
        elif word in NEGATIVE:
            score = -1
        else:
            continue

        previous = {
            w.strip(".,!?;:\"'()[]{}")
            for w in words[max(0, i - 3):i]
        }
        if previous & NEGATIONS:
            score *= -1

        if score > 0:
            positive_score += 1
        else:
            negative_score += 1

    if positive_score > negative_score:
        return "happy"
    if negative_score > positive_score:
        return "sad"
    return "neutral"

@app.get("/")
def root():
    return {"message": "Batch Sentiment Analysis API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

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
