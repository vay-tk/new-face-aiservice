from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic.json import pydantic_encoder
from typing import List, Optional
import uvicorn
import os
import json
from dotenv import load_dotenv
from face_recognition_service import FaceRecognitionService

load_dotenv()

app = FastAPI(title="Face Recognition & Liveness Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognition service
face_service = FaceRecognitionService()

# Request models
class EnrollRequest(BaseModel):
    image: str  # Base64 encoded image
    imageId: str

class LivenessRequest(BaseModel):
    frames: List[str]  # List of base64 encoded frames

class MatchRequest(BaseModel):
    probeImage: str  # Base64 encoded image
    userEmbeddings: List[List[float]]  # List of user embeddings
    userId: str

class MatchWithLivenessRequest(BaseModel):
    frames: List[str]  # List of base64 encoded frames for liveness
    probeImage: str  # Base64 encoded probe image
    userEmbeddings: List[List[float]]  # List of user embeddings
    userId: str

# Response models
class EnrollResponse(BaseModel):
    success: bool
    embedding: Optional[List[float]] = None
    imageId: Optional[str] = None
    error: Optional[str] = None

class LivenessResponse(BaseModel):
    success: bool
    isLive: bool
    score: float
    blinkDetected: bool
    headMovementDetected: bool
    details: dict
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            # Handle numpy types
            'numpy.bool_': bool,
            'numpy.int64': int,
            'numpy.float64': float,
        }

class MatchResponse(BaseModel):
    success: bool
    isMatch: bool
    confidence: float
    bestMatchIndex: int
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            'numpy.bool_': bool,
            'numpy.int64': int,
            'numpy.float64': float,
        }

class MatchWithLivenessResponse(BaseModel):
    success: bool
    livenessResult: dict
    matchResult: dict
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            'numpy.bool_': bool,
            'numpy.int64': int,
            'numpy.float64': float,
        }

@app.get("/")
async def root():
    return {"message": "Face Recognition & Liveness Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "face-recognition-ai"}

@app.post("/enroll", response_model=EnrollResponse)
async def enroll_face(request: EnrollRequest):
    """
    Extract face embedding from a single image for enrollment
    """
    try:
        result = face_service.extract_embedding(request.image)
        
        if result["success"]:
            return EnrollResponse(
                success=True,
                embedding=result["embedding"],
                imageId=request.imageId
            )
        else:
            return EnrollResponse(
                success=False,
                error=result["error"]
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@app.post("/liveness", response_model=LivenessResponse)
async def detect_liveness(request: LivenessRequest):
    """
    Perform active liveness detection on a sequence of frames
    """
    try:
        if len(request.frames) < 6:
            return LivenessResponse(
                success=False,
                isLive=False,
                score=0.0,
                blinkDetected=False,
                headMovementDetected=False,
                details={},
                error="Insufficient frames for liveness detection (minimum 6 required)"
            )
        
        result = face_service.detect_liveness(request.frames)
        
        return LivenessResponse(
            success=result["success"],
            isLive=result["isLive"],
            score=result["score"],
            blinkDetected=result["blinkDetected"],
            headMovementDetected=result["headMovementDetected"],
            details=result["details"],
            error=result.get("error")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liveness detection failed: {str(e)}")

@app.post("/match", response_model=MatchResponse)
async def match_face(request: MatchRequest):
    """
    Match a probe image against stored embeddings
    """
    try:
        if not request.userEmbeddings:
            return MatchResponse(
                success=False,
                isMatch=False,
                confidence=0.0,
                bestMatchIndex=-1,
                error="No enrolled embeddings found for user"
            )
        
        result = face_service.match_face(request.probeImage, request.userEmbeddings)
        
        return MatchResponse(
            success=result["success"],
            isMatch=result["isMatch"],
            confidence=result["confidence"],
            bestMatchIndex=result["bestMatchIndex"],
            error=result.get("error")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face matching failed: {str(e)}")

@app.post("/match-with-liveness", response_model=MatchWithLivenessResponse)
async def match_with_liveness(request: MatchWithLivenessRequest):
    """
    Perform both liveness detection and face matching in a single request
    """
    try:
        # First, perform liveness detection
        liveness_result = face_service.detect_liveness(request.frames)
        
        # Then, perform face matching
        match_result = face_service.match_face(request.probeImage, request.userEmbeddings)
        
        return MatchWithLivenessResponse(
            success=liveness_result["success"] and match_result["success"],
            livenessResult={
                "isLive": liveness_result["isLive"],
                "score": liveness_result["score"],
                "blinkDetected": liveness_result["blinkDetected"],
                "headMovementDetected": liveness_result["headMovementDetected"],
                "details": liveness_result["details"]
            },
            matchResult={
                "isMatch": match_result["isMatch"],
                "confidence": match_result["confidence"],
                "bestMatchIndex": match_result["bestMatchIndex"]
            },
            error=liveness_result.get("error") or match_result.get("error")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined detection failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )