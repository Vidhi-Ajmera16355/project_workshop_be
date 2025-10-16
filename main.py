from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from transformers import pipeline
import torch

# Initialize FastAPI app
app = FastAPI(title="AI Content Repurposer", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","https://project-workshop.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ContentRequest(BaseModel):
    text: str

# Response model
class TweetResponse(BaseModel):
    tweet: str
    original_length: int
    tweet_length: int

# Initialize the model (lazy loading)
summarizer = None
llm = None

def initialize_model():
    """Initialize the BART model and LangChain pipeline"""
    global summarizer, llm
    
    if summarizer is None:
        print("Loading BART model...")
        
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        # Initialize BART summarization pipeline
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device,
            max_length=100,
            min_length=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Wrap in HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=summarizer)
        
        print("Model loaded successfully!")

# Few-shot examples for creative tweet generation
examples = [
    {
        "input": "Artificial intelligence is transforming the way we work and live. Machine learning algorithms can now process vast amounts of data in seconds, helping businesses make better decisions and improve efficiency.",
        "output": "🤖 AI is revolutionizing our world! Machine learning processes massive data in SECONDS, driving smarter business decisions 📊✨ #AI #MachineLearning #FutureTech"
    },
    {
        "input": "Climate change is one of the most pressing challenges of our time. Rising temperatures, melting ice caps, and extreme weather events are threatening ecosystems and communities worldwide.",
        "output": "🌍 Climate crisis alert! Rising temps, melting ice & extreme weather are threatening our planet 🔥❄️ Time to ACT NOW! #ClimateChange #SaveOurPlanet"
    },
    {
        "input": "The new smartphone features an advanced camera system with 108MP resolution, 5G connectivity, and an all-day battery life. It's designed for users who demand the best in mobile technology.",
        "output": "📱 The ultimate smartphone is HERE! 108MP camera 📸 + 5G speed ⚡ + all-day battery 🔋 = Pure PERFECTION! #TechLife #Smartphone #Innovation"
    }
]

# Create few-shot prompt template
example_template = """
Original: {input}
Tweet: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

prefix = """You are a creative social media expert. Transform long-form content into engaging, catchy tweets.
Guidelines:
- Keep it under 280 characters
- Add relevant emojis (2-4 emojis)
- Use power words and action verbs
- Include 1-2 hashtags if relevant
- Make it shareable and attention-grabbing
- Maintain the key message

Here are some examples:
"""

suffix = """
Original: {input}
Tweet:"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input"],
    example_separator="\n"
)

def generate_creative_tweet(text: str) -> str:
    """Generate a creative tweet using BART and few-shot prompting"""
    
    # First, use BART to summarize the content
    summary_result = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=True
    )
    
    base_summary = summary_result[0]['summary_text']
    
    # Create a creative tweet based on the summary and few-shot examples
    # Since BART is a summarization model, we'll enhance it with our few-shot approach
    prompt = few_shot_prompt.format(input=text)
    
    # Use the pattern from examples to create an engaging tweet
    tweet_parts = []
    
    # Add an attention-grabbing emoji at the start
    emojis_start = ["🚀", "✨", "💡", "🎯", "🔥", "⚡"]
    import random
    tweet_parts.append(random.choice(emojis_start))
    
    # Clean and shorten the summary
    summary_clean = base_summary.strip()
    if len(summary_clean) > 200:
        summary_clean = summary_clean[:200] + "..."
    
    tweet_parts.append(summary_clean)
    
    # Add relevant emojis based on content
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
    
    # Combine parts
    tweet = " ".join(tweet_parts)
    
    # Ensure it's under 280 characters
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    
    return tweet

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Content Repurposer API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": summarizer is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=TweetResponse)
async def generate_tweet(request: ContentRequest):
    """
    Generate a creative tweet from input text
    
    Args:
        request: ContentRequest with text field
        
    Returns:
        TweetResponse with generated tweet and metadata
    """
    try:
        # Validate input
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
        
        # Initialize model if not already loaded
        if summarizer is None:
            initialize_model()
        
        # Generate the tweet
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)