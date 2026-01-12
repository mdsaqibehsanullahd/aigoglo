"""
RFQ Intelligence System - Main FastAPI Application
AI-powered RFQ normalization and enrichment platform
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from config import settings
from models import RFQOutput, RFQRequest
from services.extractor import extractor_service
from services.ollama_service import ollama_service
from services.normalizer import normalizer_service
from utils.helpers import generate_request_id, sanitize_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API health check and info"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "ollama_endpoint": settings.OLLAMA_BASE_URL,
        "supported_formats": [
            "text", "pdf", "excel", "csv", "json", "image", "audio"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    import httpx
    
    # Check Ollama connectivity
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                ollama_status = "connected"
            else:
                ollama_status = "error"
    except Exception as e:
        ollama_status = f"unreachable: {str(e)}"
    
    return {
        "api": "healthy",
        "ollama": ollama_status,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/rfq/text", response_model=RFQOutput)
async def process_text_rfq(request: RFQRequest):
    """
    Process text-based RFQ
    
    Input: Plain text RFQ (email, WhatsApp, etc.)
    Output: Canonical RFQ JSON
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing text RFQ")
    
    try:
        # Extract text (already provided)
        extracted_text, metadata = await extractor_service.extract_from_text(request.text)
        
        # LLM reasoning
        llm_output = await ollama_service.generate_rfq_json(
            extracted_text, 
            context=request.context
        )
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully processed text RFQ")
        return rfq_output
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing text RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rfq/pdf", response_model=RFQOutput)
async def process_pdf_rfq(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Process PDF RFQ
    
    Input: PDF file (tender documents, quotations, etc.)
    Output: Canonical RFQ JSON
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing PDF RFQ: {file.filename}")
    
    try:
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Extract text from PDF
        extracted_text, metadata = await extractor_service.extract_from_pdf(content)
        
        # LLM reasoning
        llm_output = await ollama_service.generate_rfq_json(extracted_text, context)
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully processed PDF RFQ")
        return rfq_output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error processing PDF RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rfq/excel", response_model=RFQOutput)
async def process_excel_rfq(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Process Excel/CSV RFQ
    
    Input: Excel BOM/BOQ or CSV file
    Output: Canonical RFQ JSON
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing Excel RFQ: {file.filename}")
    
    try:
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Extract text from Excel
        extracted_text, metadata = await extractor_service.extract_from_excel(
            content, 
            file.filename
        )
        
        # LLM reasoning
        llm_output = await ollama_service.generate_rfq_json(extracted_text, context)
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully processed Excel RFQ")
        return rfq_output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error processing Excel RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rfq/image", response_model=RFQOutput)
async def process_image_rfq(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Process Image RFQ
    
    Input: Scanned RFQ, photo, etc.
    Output: Canonical RFQ JSON
    Uses: llava:13b vision model
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing Image RFQ: {file.filename}")
    
    try:
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Process image
        image_data, metadata = await extractor_service.extract_from_image(content)
        
        # LLM vision processing
        llm_output = await ollama_service.process_image_rfq(image_data, context)
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully processed Image RFQ")
        return rfq_output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error processing Image RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rfq/audio", response_model=RFQOutput)
async def process_audio_rfq(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Process Audio RFQ
    
    Input: Voice RFQ recording
    Output: Canonical RFQ JSON
    Uses: karanchopda333/whisper model
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing Audio RFQ: {file.filename}")
    
    try:
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Process audio
        audio_data, metadata = await extractor_service.extract_from_audio(content)
        
        # LLM audio processing (Whisper)
        llm_output = await ollama_service.process_audio_rfq(audio_data, context)
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully processed Audio RFQ")
        return rfq_output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error processing Audio RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rfq/json", response_model=RFQOutput)
async def process_json_rfq(request: Dict[str, Any] = Body(...)):
    """
    Process raw JSON RFQ
    
    Input: Arbitrary JSON object in request body
    Output: Canonical RFQ JSON
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing JSON RFQ Input")
    
    try:
        # Dump the incoming JSON to string for the LLM
        # This handles the "removal of context wrapper" requirement
        json_text = json.dumps(request, default=str)
        
        llm_output = await ollama_service.generate_rfq_json(
            json_text, 
            context="Processed from direct JSON input"
        )
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully processed JSON RFQ")
        return rfq_output
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing JSON RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rfq/auto", response_model=RFQOutput)
async def process_auto_rfq(
    file: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Auto-detect and process RFQ
    
    Input: Any supported format
    Output: Canonical RFQ JSON
    
    This endpoint automatically detects the file type and routes to appropriate processor
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Auto-processing RFQ: {file.filename}")
    
    try:
        # Detect file type
        file_type = extractor_service.detect_file_type(file.filename, file.content_type)
        logger.info(f"[{request_id}] Detected file type: {file_type}")
        
        # Read content
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Route to appropriate extractor
        if file_type == 'pdf':
            extracted_text, metadata = await extractor_service.extract_from_pdf(content)
        elif file_type == 'excel':
            extracted_text, metadata = await extractor_service.extract_from_excel(content, file.filename)
        elif file_type == 'image':
            image_data, metadata = await extractor_service.extract_from_image(content)
            llm_output = await ollama_service.process_image_rfq(image_data, context)
            rfq_output = normalizer_service.normalize_and_validate(llm_output)
            return rfq_output
        elif file_type == 'audio':
            audio_data, metadata = await extractor_service.extract_from_audio(content)
            llm_output = await ollama_service.process_audio_rfq(audio_data, context)
            rfq_output = normalizer_service.normalize_and_validate(llm_output)
            return rfq_output
        elif file_type == 'json':
            extracted_text, metadata = await extractor_service.extract_from_json(content)
        else:
            extracted_text, metadata = await extractor_service.extract_from_text(content.decode('utf-8'))
        
        # LLM reasoning
        llm_output = await ollama_service.generate_rfq_json(extracted_text, context)
        
        # Normalize and validate
        rfq_output = normalizer_service.normalize_and_validate(llm_output)
        
        logger.info(f"[{request_id}] Successfully auto-processed RFQ")
        return rfq_output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error auto-processing RFQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
