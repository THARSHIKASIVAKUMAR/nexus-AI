import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

from services.preprocess import preprocess_file
from services.interpret import call_ai_unified
from services.predict import compute_metrics
from services.insight import extract_insights

app = FastAPI(title="NexusAI Backend")

# Configure CORS - set to "*" for development to ensure it works across all local environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str | list

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    provider: Optional[str] = "google" # Default to Google as user provided key
    system_prompt: Optional[str] = "You are NexusAI, an expert analyst."

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "providers": {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_API_KEY").startswith("your_sk")),
            "google": bool(os.getenv("GOOGLE_API_KEY")),
            "huggingface": bool(os.getenv("HUGGINGFACE_TOKEN"))
        }
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    metadata = await preprocess_file(file)
    return {
        "filename": metadata["filename"],
        "type": metadata["type"],
        "extension": metadata["extension"],
        "size_bytes": metadata["size_bytes"],
        "media_type": metadata["media_type"]
    }

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    prompt: str = Form("Please provide a comprehensive analysis of this file."),
    provider: str = Form("google")
):
    metadata = await preprocess_file(file)
    
    # Simple prompt construction for unified call
    full_prompt = prompt
    if metadata["type"] != "image":
        truncated_content = metadata["content"][:15000] if metadata["content"] else "[Empty File]"
        full_prompt = f"File: {metadata['filename']}\n\nContent:\n{truncated_content}\n\n---\n\n{prompt}"

    ai_response = await call_ai_unified(
        provider=provider,
        prompt=full_prompt,
        image_data=metadata["image_b64"] if metadata["type"] == "image" else None,
        media_type=metadata["media_type"]
    )

    metrics = compute_metrics(ai_response)
    top_insights = extract_insights(ai_response)

    return {
        "analysis": ai_response,
        "metrics": metrics,
        "insights": top_insights,
        "metadata": {
            "filename": metadata["filename"],
            "type": metadata["type"]
        },
        "provider": provider
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    # Extract last message content for unified call
    last_msg = request.messages[-1].content
    if isinstance(last_msg, list):
        # Handle list content (usually vision)
        text_parts = [p["text"] for p in last_msg if p["type"] == "text"]
        prompt = " ".join(text_parts)
    else:
        prompt = last_msg

    ai_response = await call_ai_unified(
        provider=request.provider,
        prompt=prompt,
        messages=[msg.dict() for msg in request.messages]
    )
    
    return {"content": ai_response, "provider": request.provider}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
