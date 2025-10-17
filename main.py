from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch

# Initialize FastAPI app
app = FastAPI(title="AI Content Repurposer", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ContentRequest(BaseModel):
    text: str

# Response models
class TweetResponse(BaseModel):
    tweet: str
    original_length: int
    tweet_length: int

class SummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    all_scores: dict

class CompleteAnalysisResponse(BaseModel):
    tweet: str
    summary: str
    sentiment: str
    sentiment_confidence: float
    original_length: int

# Initialize models (lazy loading)
summarizer = None
sentiment_analyzer = None

def initialize_models():
    """Initialize all models"""
    global summarizer, sentiment_analyzer
    
    if summarizer is None:
        print("Loading models...")
        device = 0 if torch.cuda.is_available() else -1
        
        # BART for summarization
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
            max_length=130,
            min_length=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # DistilBERT for sentiment analysis
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
        
        print("Models loaded successfully!")

def generate_creative_tweet(text: str) -> str:
    """Generate a creative tweet using BART"""
    import random
    
    # Summarize the content
    summary_result = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=True
    )
    
    base_summary = summary_result[0]['summary_text']
    
    # Start with an attention-grabbing emoji
    emojis_start = ["🚀", "✨", "💡", "🎯", "🔥", "⚡"]
    tweet_parts = [random.choice(emojis_start)]
    
    # Clean summary
    summary_clean = base_summary.strip()
    if len(summary_clean) > 200:
        summary_clean = summary_clean[:200] + "..."
    
    tweet_parts.append(summary_clean)
    
    # Add contextual emojis
    content_lower = text.lower()
    if any(word in content_lower for word in ["tech", "ai", "digital", "innovation"]):
        tweet_parts.append("💻🤖")
    elif any(word in content_lower for word in ["business", "success", "growth"]):
        tweet_parts.append("📈💼")
    elif any(word in content_lower for word in ["health", "wellness", "fitness"]):
        tweet_parts.append("💪🌟")
    elif any(word in content_lower for word in ["environment", "climate", "nature"]):
        tweet_parts.append("🌍🌱")
    else:
        tweet_parts.append("✨")
    
    tweet = " ".join(tweet_parts)
    
    # Ensure under 280 characters
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    
    return tweet

def generate_summary(text: str, max_length: int = 130) -> str:
    """Generate a summary of the text"""
    result = summarizer(
        text,
        max_length=max_length,
        min_length=30,
        do_sample=False
    )
    return result[0]['summary_text']

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of the text"""
    result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
    return result

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Content Repurposer API",
        "status": "running",
        "version": "2.0.0",
        "features": ["tweet_generation", "summarization", "sentiment_analysis"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": summarizer is not None and sentiment_analyzer is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=TweetResponse)
async def generate_tweet(request: ContentRequest):
    """Generate a creative tweet from input text"""
    try:
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Input text is too short. Please provide at least 10 characters."
            )
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Input text is too long. Please keep it under 5000 characters."
            )
        
        if summarizer is None:
            initialize_models()
        
        tweet = generate_creative_tweet(request.text)
        
        return TweetResponse(
            tweet=tweet,
            original_length=len(request.text),
            tweet_length=len(tweet)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating tweet: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate tweet: {str(e)}"
        )

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: ContentRequest):
    """Generate a summary of the input text"""
    try:
        if not request.text or len(request.text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Input text is too short. Please provide at least 50 characters for summarization."
            )
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Input text is too long. Please keep it under 5000 characters."
            )
        
        if summarizer is None:
            initialize_models()
        
        summary = generate_summary(request.text)
        
        return SummaryResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            compression_ratio=round(len(summary) / len(request.text), 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: ContentRequest):
    """Analyze the sentiment of the input text"""
    try:
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Input text is too short. Please provide at least 10 characters."
            )
        
        if sentiment_analyzer is None:
            initialize_models()
        
        sentiment_result = analyze_sentiment(request.text)
        
        return SentimentResponse(
            sentiment=sentiment_result['label'],
            confidence=round(sentiment_result['score'], 4),
            all_scores={
                sentiment_result['label']: round(sentiment_result['score'], 4)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze sentiment: {str(e)}"
        )

@app.post("/analyze-all", response_model=CompleteAnalysisResponse)
async def complete_analysis(request: ContentRequest):
    """Perform complete analysis: tweet generation, summarization, and sentiment analysis"""
    try:
        if not request.text or len(request.text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Input text is too short. Please provide at least 50 characters."
            )
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Input text is too long. Please keep it under 5000 characters."
            )
        
        if summarizer is None or sentiment_analyzer is None:
            initialize_models()
        
        # Generate all analyses
        tweet = generate_creative_tweet(request.text)
        summary = generate_summary(request.text)
        sentiment_result = analyze_sentiment(request.text)
        
        return CompleteAnalysisResponse(
            tweet=tweet,
            summary=summary,
            sentiment=sentiment_result['label'],
            sentiment_confidence=round(sentiment_result['score'], 4),
            original_length=len(request.text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in complete analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform complete analysis: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)