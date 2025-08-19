from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import gc
import logging
import json
import zipfile
from pathlib import Path
from typing import List, Optional
import torch
from pydantic import BaseModel

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.default import Config
from src.inference import EuroSATClassifier

# Response models
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict
    top_classes: List[tuple]
    success: bool
    error: Optional[str] = None

class BatchResult(BaseModel):
    results: List[dict]
    summary: dict

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str

# Initialize FastAPI app
app = FastAPI(
    title="EuroSAT Land Cover Classification API",
    description="REST API for satellite image land cover classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model_path = Config.TRAINED_MODEL_DIR / 'best_model.pth'
        if not model_path.exists():
            logging.error("Model file not found")
            return
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = EuroSATClassifier(str(model_path))
        logging.info("Model loaded successfully")
        
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "EuroSAT Land Cover Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        device=str(model.device) if model else "unknown",
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResult)
async def predict_single(
    file: UploadFile = File(...),
    use_tta: bool = False,
    return_gradcam: bool = False
):
    """Predict single image"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp_uploads") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Make prediction
        result = model.predict_image(
            str(temp_path), 
            use_tta=use_tta, 
            return_gradcam=return_gradcam
        )
        
        # Clean up
        temp_path.unlink()
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Prediction failed'))
        
        return PredictionResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(
    files: List[UploadFile] = File(...),
    use_tta: bool = False,
    confidence_threshold: float = 0.5
):
    """Predict multiple images"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 100 allowed.")
    
    results = []
    temp_files = []
    
    try:
        # Process each file
        for file in files:
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Invalid file type'
                })
                continue
            
            # Save file temporarily
            temp_path = Path("temp_uploads") / file.filename
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            temp_files.append(temp_path)
            
            # Make prediction
            result = model.predict_image(str(temp_path), use_tta=use_tta)
            result['filename'] = file.filename
            result['meets_threshold'] = result['confidence'] >= confidence_threshold if result['success'] else False
            results.append(result)
        
        # Generate summary
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        above_threshold = sum(1 for r in results if r.get('meets_threshold', False))
        
        summary = {
            'total_files': total,
            'successful_predictions': successful,
            'above_threshold': above_threshold,
            'success_rate': successful / total if total > 0 else 0
        }
        
        return BatchResult(results=results, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            if temp_path.exists():
                temp_path.unlink()

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    return {
        "classes": Config.CLASSES,
        "count": len(Config.CLASSES)
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Load training metadata if available
    metadata_path = Config.TRAINED_MODEL_DIR / 'training_metadata.json'
    training_info = {}
    
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                training_info = json.load(f)
        except:
            pass
    
    return {
        "architecture": "MobileNetV3-Large",
        "classes": Config.CLASSES,
        "device": str(model.device),
        "input_size": Config.IMAGE_SIZE,
        "training_info": training_info
    }

# Background task for cleanup
def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    temp_dir = Path("temp_uploads")
    if not temp_dir.exists():
        return
    
    import time
    current_time = time.time()
    for file_path in temp_dir.glob("*"):
        try:
            if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:
                file_path.unlink()
        except:
            pass

@app.post("/admin/cleanup")
async def manual_cleanup(background_tasks: BackgroundTasks):
    """Manually trigger cleanup of temporary files"""
    background_tasks.add_task(cleanup_temp_files)
    return {"message": "Cleanup task started"}
