"""
RFQ Intelligence System - Main FastAPI Application
AI-powered RFQ normalization and enrichment platform
"""
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from typing import List, Optional, Dict, Any
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
async def process_auto_rfq_multiple(
    files: List[UploadFile] = File(...),
    context: Optional[str] = Form(None)
):
    """
    Process multiple RFQ files and return ALL products from ALL files
    Combines products from: PDFs + Excel + Images + Audio
    """
    request_id = generate_request_id()
    logger.info(f"[{request_id}] Processing multiple files for multi-product output: {[f.filename for f in files]}")
    
    try:
        all_products = []
        combined_details = {}
        total_confidence = 0
        file_count = 0
        
        # Process each file individually first
        for file in files:
            logger.info(f"[{request_id}] Processing file: {file.filename}")
            
            # Validate file size
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"[{request_id}] File {file.filename} too large, skipping")
                continue
            
            try:
                # Detect file type
                file_type = extractor_service.detect_file_type(file.filename, file.content_type)
                
                # Extract content based on file type
                if file_type == 'pdf':
                    extracted_text, metadata = await extractor_service.extract_from_pdf(content)
                elif file_type == 'excel':
                    extracted_text, metadata = await extractor_service.extract_from_excel(content, file.filename)
                elif file_type == 'image':
                    image_data, metadata = await extractor_service.extract_from_image(content)
                    # Process image separately
                    llm_output = await ollama_service.process_image_rfq(image_data, f"File: {file.filename}")
                    if llm_output and 'products' in llm_output:
                        all_products.extend(llm_output['products'])
                        total_confidence += llm_output.get('confidence_score', 50)
                        file_count += 1
                    continue
                elif file_type == 'audio':
                    audio_data, metadata = await extractor_service.extract_from_audio(content)
                    audio_text = await ollama_service.process_audio_rfq(audio_data, f"File: {file.filename}")
                    if isinstance(audio_text, dict) and 'products' in audio_text:
                        all_products.extend(audio_text['products'])
                        total_confidence += audio_text.get('confidence_score', 50)
                        file_count += 1
                    continue
                else:
                    extracted_text, metadata = await extractor_service.extract_from_text(content.decode('utf-8'))
                
                # Process extracted text for this file
                if extracted_text.strip():
                    file_context = f"File: {file.filename} | {context or ''}"
                    llm_output = await ollama_service.generate_rfq_json(extracted_text, file_context)
                    
                    if llm_output and 'products' in llm_output:
                        # Add filename to each product for tracking
                        for product in llm_output['products']:
                            if 'metadata' not in product:
                                product['metadata'] = {}
                            product['metadata']['source_file'] = file.filename
                        
                        all_products.extend(llm_output['products'])
                        total_confidence += llm_output.get('confidence_score', 50)
                        file_count += 1
                        
                        # Merge commercial details (take first non-null values)
                        details = llm_output.get('extracted_details', {})
                        for key, value in details.items():
                            if value is not None and key not in combined_details:
                                combined_details[key] = value
                    
            except Exception as e:
                logger.error(f"[{request_id}] Error processing individual file {file.filename}: {e}")
                # Continue with other files even if one fails
                continue
        
        # If no products found, create fallback
        if not all_products:
            logger.warning(f"[{request_id}] No products found in any file, creating fallback")
            fallback_output = create_multi_file_fallback(files, context)
            return normalizer_service.normalize_and_validate(fallback_output)
        
        # Calculate average confidence
        avg_confidence = total_confidence // max(file_count, 1)
        
        # Create combined output
        combined_output = {
            "confidence_score": min(avg_confidence, 95),  # Cap at 95 for multi-file
            "accuracy": determine_multi_file_accuracy(all_products),
            "rfq_archetype": determine_multi_file_archetype(all_products),
            "products": all_products,
            "extracted_details": combined_details if combined_details else None,
            "Ai_generated": create_multi_file_ai_summary(all_products, files, context)
        }
        
        # Normalize and validate final output
        rfq_output = normalizer_service.normalize_and_validate(combined_output)
        
        logger.info(f"[{request_id}] Successfully combined {len(all_products)} products from {file_count} files")
        return rfq_output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in multi-file processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions (move outside the main function)
def create_multi_file_fallback(files, context):
    """Create fallback response for multi-file processing"""
    return {
        "confidence_score": 30,
        "accuracy": "low",
        "rfq_archetype": "Item RFQ (single or multi-line products)",
        "products": [
            {
                "product_name": "Multi-File RFQ Processing",
                "product_type": "General",
                "qty": len(files),
                "unit": "files",
                "specifications": f"Processed {len(files)} files: {[f.filename for f in files]}",
                "status": "need_clarity",
                "Model": "MDL-MUL-001",
                "Category": "General",
                "Estimated_cost": "TBD",
                "preference": "OEM/Equivalent",
                "Bom": "BOM-2024-MUL",
                "Sku": "SKU-MUL-000001",
                "Hs_code": "0000.00.00",
                "line_confidence": 0.3
            }
        ],
        "extracted_details": None,
        "Ai_generated": {
            "ai_suggestion": "Multiple files uploaded but content unclear. Please ensure files contain clear product specifications.",
            "tag": "multi_file_processing",
            "category": "General",
            "describe_request": f"Multi-file RFQ from {len(files)} files requiring clarification",
            "Technical_drawning": "Not available",
            "total_value": "TBD",
            "currency": "USD",
            "target_region": "Global"
        }
    }

def determine_multi_file_accuracy(products):
    """Determine accuracy based on total products and completeness"""
    if not products:
        return "low"
    
    total_fields = sum(len([k for k in prod.keys() if prod[k] is not None]) for prod in products)
    avg_fields = total_fields // max(len(products), 1)
    
    if avg_fields > 12:
        return "high"
    elif avg_fields >= 8:
        return "medium"
    else:
        return "low"

def determine_multi_file_archetype(products):
    """Determine archetype from multiple products"""
    if not products:
        return "Item RFQ (single or multi-line products)"
    
    if len(products) > 10:
        return "BOM/BOQ RFQ (Excel list of items for projects)"
    elif len(products) > 5:
        return "Item RFQ (single or multi-line products)"
    elif any("system" in prod.get("product_name", "").lower() for prod in products):
        return "System/Package RFQ (complete skid/system/line)"
    else:
        return "Item RFQ (single or multi-line products)"

def create_multi_file_ai_summary(products, files, context):
    """Create AI summary for multi-file processing"""
    total_products = len(products)
    total_quantity = sum(prod.get("qty", 0) for prod in products)
    categories = list(set(prod.get("Category", "General") for prod in products))
    
    # Calculate total value
    total_value = 0
    for prod in products:
        cost_str = str(prod.get("Estimated_cost", "0"))
        numbers = re.findall(r'\d+', cost_str)
        if numbers:
            unit_cost = int(numbers[0])
            total_value += unit_cost * prod.get("qty", 1)
    
    return {
        "ai_suggestion": f"Multiple files processed containing {total_products} different products. Consider bulk purchasing for potential volume discounts.",
        "tag": "multi_file_procurement",
        "category": categories[0] if len(categories) == 1 else "Mixed Equipment",
        "describe_request": f"Combined RFQ from {len(files)} files containing {total_products} products with total quantity of {total_quantity} units",
        "Technical_drawning": "Available upon request",
        "total_value": f"{total_value:,}",
        "currency": "USD",
        "target_region": "Global"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
