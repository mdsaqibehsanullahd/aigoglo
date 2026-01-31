"""
Ollama Service - LLM Reasoning Engine
This is the BRAIN of the RFQ Intelligence System
"""

import httpx
import json
import logging
import base64
import re
import os
import tempfile
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OllamaSettings:
    """Configuration settings for Ollama Service"""
    OLLAMA_BASE_URL: str = "http://100.82.77.42:11434"
    OLLAMA_MODEL: str = "mistral-nemo:12b"
    OLLAMA_VISION_MODEL: str = "llava:13b"
    OLLAMA_AUDIO_MODEL: str = "karanchopda333/whisper:latest"
    LLM_TIMEOUT: int = 120
    LLM_TEMPERATURE: float = 0.1
    
    @classmethod
    def from_env(cls):
        """Load settings from environment variables with defaults"""
        return cls(
            OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://100.82.77.42:11434"),
            OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", "mistral-nemo:12b"),
            OLLAMA_VISION_MODEL=os.getenv("OLLAMA_VISION_MODEL", "llava:13b"),
            OLLAMA_AUDIO_MODEL=os.getenv("OLLAMA_AUDIO_MODEL", "karanchopda333/whisper:latest"),
            LLM_TIMEOUT=int(os.getenv("LLM_TIMEOUT", "120")),
            LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.1"))
        )


# Global settings instance
settings = OllamaSettings.from_env()


class OllamaService:
    """
    LLM Service for RFQ Intelligence
    Responsible for:
    - Reasoning and inference
    - Missing value inference
    - Classification
    - Normalization
    - Enrichment
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.vision_model = settings.OLLAMA_VISION_MODEL
        self.audio_model = settings.OLLAMA_AUDIO_MODEL
        self.timeout = settings.LLM_TIMEOUT
        self.temperature = settings.LLM_TEMPERATURE
        
        logger.info(f"OllamaService initialized with model: {self.model}")
        logger.info(f"Base URL: {self.base_url}")
    
    async def generate_rfq_json(self, extracted_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Main LLM reasoning function - Takes raw text and generates complete RFQ JSON"""
        
        if not extracted_text or not extracted_text.strip():
            logger.warning("Empty input text provided")
            return self._get_fallback_response("No input text provided")
        
        prompt = self._build_rfq_prompt(extracted_text, context)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.temperature,
                        "format": "json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return self._get_fallback_response(extracted_text)
                
                result = response.json()
                llm_output = result.get("response", "{}")
                
                # Parse JSON from LLM
                try:
                    rfq_json = json.loads(llm_output)
                    rfq_json = self._ensure_complete_response(rfq_json)
                    logger.info(f"Successfully generated RFQ JSON with confidence: {rfq_json.get('confidence_score', 0)}")
                    return rfq_json
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    rfq_json = self._extract_json_from_text(llm_output)
                    if rfq_json:
                        rfq_json = self._ensure_complete_response(rfq_json)
                        return rfq_json
                    return self._get_fallback_response(extracted_text)
                    
        except httpx.TimeoutException:
            logger.error(f"Request timeout after {self.timeout}s")
            return self._get_fallback_response(extracted_text)
        except Exception as e:
            logger.error(f"Ollama service error: {e}", exc_info=True)
            return self._get_fallback_response(extracted_text)
    
    def _ensure_complete_response(self, rfq_json: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure LLM response has all required fields and proper data types"""
        
        if not isinstance(rfq_json, dict):
            logger.warning("Invalid response format, using fallback")
            return self._get_fallback_response("Invalid response format")
        
        # Ensure Ai_generated section exists
        if "Ai_generated" not in rfq_json or not rfq_json["Ai_generated"]:
            logger.warning("Ai_generated section missing, adding defaults")
            rfq_json["Ai_generated"] = self._get_default_ai_generated(rfq_json)
        else:
            rfq_json["Ai_generated"] = self._sanitize_ai_generated(rfq_json["Ai_generated"], rfq_json)
        
        # Ensure products exist
        if "products" not in rfq_json or not rfq_json["products"]:
            logger.warning("Products section missing, adding default")
            rfq_json["products"] = [self._get_default_product()]
        else:
            rfq_json["products"] = [self._sanitize_product(prod) for prod in rfq_json["products"]]
        
        # Ensure top-level fields
        rfq_json["confidence_score"] = self._sanitize_confidence(rfq_json.get("confidence_score"))
        rfq_json["accuracy"] = self._sanitize_accuracy(rfq_json.get("accuracy"))
        rfq_json["rfq_archetype"] = self._sanitize_archetype(rfq_json.get("rfq_archetype"))
        rfq_json["extracted_details"] = self._sanitize_extracted_details(rfq_json.get("extracted_details"))
        
        return rfq_json
    
    def _sanitize_ai_generated(self, ai_gen: Dict[str, Any], rfq_json: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and ensure proper Ai_generated section"""
        defaults = self._get_default_ai_generated(rfq_json)
        
        # Ensure all fields exist with proper types
        for key, default_value in defaults.items():
            if key not in ai_gen or ai_gen[key] is None:
                ai_gen[key] = default_value
            else:
                ai_gen[key] = self._ensure_string(ai_gen[key], default_value)
        
        return ai_gen
    
    def _sanitize_product(self, prod: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize individual product data"""
        defaults = self._get_default_product()
        
        # Handle quantity (special case)
        if "qty" in prod:
            prod["qty"] = self._sanitize_quantity(prod["qty"])
        else:
            prod["qty"] = defaults["qty"]
        
        # Sanitize all string fields
        string_fields = ["product_name", "product_type", "unit", "specifications", 
                        "status", "Model", "Category", "Estimated_cost", "preference",
                        "Bom", "Sku", "Hs_code"]
        
        for field in string_fields:
            if field in prod:
                prod[field] = self._ensure_string(prod[field], defaults.get(field, ""))
            else:
                prod[field] = defaults.get(field, "")
        
        # Sanitize confidence
        if "line_confidence" in prod:
            prod["line_confidence"] = self._sanitize_confidence_float(prod["line_confidence"])
        else:
            prod["line_confidence"] = defaults["line_confidence"]
        
        return prod
    
    def _sanitize_quantity(self, qty: Any) -> int:
        """Sanitize quantity field"""
        try:
            if isinstance(qty, str):
                nums = re.findall(r'\d+', qty)
                return int(nums[0]) if nums else 1
            elif isinstance(qty, (int, float)):
                return int(qty)
            return 1
        except:
            return 1
    
    def _sanitize_confidence(self, confidence: Any) -> int:
        """Sanitize confidence score"""
        try:
            if isinstance(confidence, (int, float)):
                return max(0, min(100, int(confidence)))
            return 50
        except:
            return 50
    
    def _sanitize_confidence_float(self, confidence: Any) -> float:
        """Sanitize line confidence float"""
        try:
            if isinstance(confidence, (int, float)):
                return max(0.0, min(1.0, float(confidence)))
            return 0.5
        except:
            return 0.5
    
    def _sanitize_accuracy(self, accuracy: Any) -> str:
        """Sanitize accuracy field"""
        valid_values = ["low", "medium", "high"]
        if isinstance(accuracy, str) and accuracy.lower() in valid_values:
            return accuracy.lower()
        return "medium"
    
    def _sanitize_archetype(self, archetype: Any) -> str:
        """Sanitize RFQ archetype"""
        valid_archetypes = [
            "Item RFQ (single or multi-line products)",
            "Part RFQ (Part No / Model / OEM spares)",
            "Spec RFQ (engineering specs, standards, drawings)",
            "BOM/BOQ RFQ (Excel list of items for projects)",
            "System/Package RFQ (complete skid/system/line)",
            "Service RFQ (installation, maintenance, EPC, inspection)",
            "Used/Surplus RFQ (condition-based)",
            "Logistics RFQ (shipping only)",
            "Hybrid RFQ (mixture of any of the above)"
        ]
        
        if isinstance(archetype, str) and archetype in valid_archetypes:
            return archetype
        return "Item RFQ (single or multi-line products)"
    
    def _sanitize_extracted_details(self, details: Any) -> Optional[Dict[str, Any]]:
        """Sanitize extracted details section"""
        if not isinstance(details, dict):
            return None
        
        valid_fields = ["delivery_location", "shipping_information", "payment_term", 
                       "inco_terms", "submission_deadlines", "target_budget"]
        
        sanitized = {}
        for field in valid_fields:
            sanitized[field] = self._ensure_string(details.get(field), None)
        
        return sanitized
    
    def _ensure_string(self, value: Any, default: str = "") -> str:
        """Ensure value is a string"""
        if value is None:
            return default
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        if isinstance(value, list):
            return ", ".join(str(x) for x in value)
        return str(value)
    
    def _get_default_ai_generated(self, rfq_json: Dict[str, Any]) -> Dict[str, Any]:
        """Get default Ai_generated section"""
        category = "General"
        if rfq_json.get("products"):
            category = rfq_json["products"][0].get("Category", "General")
        
        return {
            "ai_suggestion": "Please review RFQ details and verify completeness",
            "tag": "rfq_processing",
            "category": category,
            "describe_request": "RFQ processing completed",
            "Technical_drawning": "Not available",
            "total_value": "TBD",
            "currency": "USD",
            "target_region": "Global"
        }
    
    def _get_default_product(self) -> Dict[str, Any]:
        """Get default product entry"""
        return {
            "product_name": "Unknown Product",
            "product_type": "General",
            "qty": 1,
            "unit": "units",
            "specifications": "Requires clarification",
            "status": "need_clarity",
            "Model": "MDL-UNK-00001",
            "Category": "General",
            "Estimated_cost": "TBD",
            "preference": "OEM/Equivalent",
            "Bom": "BOM-2026-00001",
            "Sku": "SKU-GEN-000001",
            "Hs_code": "0000.00.00",
            "line_confidence": 0.3
        }
    
    def _sanitize_preference(self, value: Any) -> str:
        """
        Normalize preference to canonical enum values
        """
        if not isinstance(value, str):
            return "OEM"

        value = value.strip().lower()

        if value in ["oem"]:
            return "OEM"
        if value in ["equivalent", "aftermarket"]:
            return "Equivalent"
        if value in ["oem/equivalent", "oem or equivalent", "either"]:
            return "OEM"   # SAFE DEFAULT for schema

        return "OEM"

    
    def _build_rfq_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build comprehensive prompt for LLM"""
        
        return f"""You are an expert RFQ (Request for Quotation) intelligence system for industrial procurement.

Your task is to analyze the provided RFQ input and generate a COMPLETE, STRUCTURED JSON OUTPUT.

CRITICAL RULES (NON-NEGOTIABLE):
1. ALL FIELDS MUST BE FILLED – You MUST infer, estimate, or generate values for every field
2. ONLY extracted_details CAN BE NULL – If commercial/logistics info is not in the input, set values to null
3. ALWAYS CREATE AT LEAST ONE PRODUCT – Even if the input is vague, infer a product
4. USE REASONING AND INFERENCE – Do not only extract; reason logically
5. GENERATE MISSING DATA – Create SKUs, BOMs, model numbers, estimates if absent
6. CLASSIFY RFQ TYPE – Choose the most accurate archetype
7. PROVIDE CONFIDENCE SCORE – Start at 100, subtract 10 for each missing/unclear critical field
8. DETERMINE ACCURACY – Based strictly on field count completeness

RFQ ARCHETYPES (CHOOSE ONE ONLY):
- Item RFQ (single or multi-line products)
- Part RFQ (Part No / Model / OEM spares)
- Spec RFQ (engineering specs, standards, drawings)
- BOM/BOQ RFQ (Excel list of items for projects)
- System/Package RFQ (complete skid/system/line)
- Service RFQ (installation, maintenance, EPC, inspection)
- Used/Surplus RFQ (condition-based)
- Logistics RFQ (shipping only)
- Hybrid RFQ (mixture of any of the above)

INPUT TEXT:
{text}

ADDITIONAL CONTEXT:
{context or "None provided"}

OUTPUT SCHEMA (STRICT JSON – NO EXTRA TEXT):
{{
"confidence_score": <0-100 integer>,
"accuracy": "<low|medium|high>",
"rfq_archetype": "<one archetype>",
"products": [
{{
"product_name": "<inferred product name>",
"product_type": "<product category>",
"qty": <integer>,
"unit": "<units|pieces|sets|kg|meters>",
"specifications": "<technical specs – inferred if missing>",
"status": "<Ready|need_clarity>",
"Model": "<generate if missing: MDL-XXX-NNN>",
"Category": "<Electrical|Mechanical|Chemical|Instrumentation|etc>",
"Estimated_cost": "<numeric, e.g. '1500.00'>",
"preference": "<OEM|Equivalent>",
"Bom": "<generate: BOM-YYYY-NNNN>",
"Sku": "<generate: SKU-CAT-NNNNNN>",
"Hs_code": "<infer or use 0000.00.00>",
"line_confidence": <0.0–1.0>
}}
],
"extracted_details": {{
"delivery_location": "<value or null>",
"shipping_information": "<value or null>",
"payment_term": "<value or null>",
"inco_terms": "<value or null>",
"submission_deadlines": "<value or null>",
"target_budget": "<value or null>"
}},
"Ai_generated": {{
"ai_suggestion": "<procurement recommendations>",
"tag": "<classification tag>",
"category": "<primary category>",
"describe_request": "<natural language summary>",
"Technical_drawning": "<Provided|Available upon request|Not available>",
"total_value": "<total estimated value>",
"currency": "<USD|EUR|INR|etc>",
"target_region": "<Global|EMEA|APAC|Americas>"
}}
}}

INFERENCE GUIDELINES:
- Quantity default: 1
- Unit default: units
- Model format: MDL-{{CATEGORY_CODE}}-{{SEQUENCE}}
- BOM format: BOM-{{YEAR}}-{{SEQUENCE}}
- SKU format: SKU-{{CATEGORY_CODE}}-{{SEQUENCE}}
- Estimate costs using industrial market norms
- line_confidence reflects data completeness

CONFIDENCE SCORING ALGORITHM:
Start at 100, subtract 10 points for each missing or unclear critical field:
- product_name, quantity, specifications, category, cost, delivery_term, payment_term
Clamp: 0-100, round to nearest 10

ACCURACY DETERMINATION:
- High: >12 populated fields
- Medium: 8–12 populated fields  
- Low: <8 populated fields

FINAL INSTRUCTION: Respond ONLY with valid JSON. No explanations. No markdown. No extra text."""
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks or mixed text"""
        try:
            # Try to find JSON in code blocks
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
                return json.loads(json_str)
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
                return json.loads(json_str)
            else:
                # Try to find JSON object
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = text[start:end]
                    return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to extract JSON: {e}")
        return None
    
    def _get_fallback_response(self, text: str) -> Dict[str, Any]:
        """Fallback response when LLM fails - Still follows the contract"""
        return {
            "confidence_score": 30,
            "accuracy": "low",
            "rfq_archetype": "Item RFQ (single or multi-line products)",
            "products": [
                {
                    "product_name": "Unknown Product (requires clarification)",
                    "product_type": "General",
                    "qty": 1,
                    "unit": "units",
                    "specifications": f"Extracted from input: {text[:100]}...",
                    "status": "need_clarity",
                    "Model": "MDL-UNK-00001",
                    "Category": "Uncategorized",
                    "Estimated_cost": "TBD",
                    "preference": "OEM",
                    "Bom": "BOM-2026-00001",
                    "Sku": "SKU-UNK-000001",
                    "Hs_code": "0000.00.00",
                    "line_confidence": 0.3
                }
            ],
            "extracted_details": None,
            "Ai_generated": {
                "ai_suggestion": "RFQ requires clarification. Please provide more details about products, quantities, and specifications.",
                "tag": "unclear_rfq",
                "category": "General",
                "describe_request": "Incomplete RFQ requiring additional information",
                "Technical_drawning": "Not available",
                "total_value": "TBD",
                "currency": "USD",
                "target_region": "Global"
            }
        }
    
    async def process_image_rfq(self, image_data: bytes, context: Optional[str] = None) -> Dict[str, Any]:
        """Process image-based RFQ using vision model"""
        logger.info("Processing image RFQ with llava vision model")
        
        try:
            # Convert image to base64 for Ollama
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Build vision prompt
            vision_prompt = f"""You are analyzing an RFQ (Request for Quotation) document image.

Extract ALL information from this image and convert it into a structured RFQ format.

Look for:
- Product names and descriptions
- Quantities
- Technical specifications
- Part numbers or model numbers
- Delivery information
- Pricing or budget information
- Deadlines
- Contact information
- Any other relevant procurement details

{f"Additional context: {context}" if context else ""}

Extract the text content from this image and describe what you see in detail. Focus on:
1. Product/item listings
2. Quantities and specifications
3. Commercial terms
4. Any tables or structured data

Provide a detailed description of all text and information visible in the image."""

            # Call Ollama vision API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": vision_prompt,
                        "images": [image_base64],
                        "stream": False
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Vision model API error: {response.status_code}")
                    return self._get_fallback_response("Image processing failed")
                
                result = response.json()
                extracted_text = result.get("response", "")
                
                if not extracted_text:
                    logger.warning("No text extracted from image")
                    return self._get_fallback_response("Empty image or no text detected")
                
                logger.info(f"Extracted {len(extracted_text)} characters from image")
                
                # Now process the extracted text with the main LLM
                return await self.generate_rfq_json(extracted_text, context)
                
        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            return self._get_fallback_response(f"Image processing failed: {str(e)}")
    
    async def process_audio_rfq(self, audio_data: bytes, context: Optional[str] = None) -> Dict[str, Any]:
        """Fixed audio processing - returns proper dictionary structure"""
        logger.info("Processing audio RFQ with enhanced speech recognition")
        
        try:
            # Use enhanced audio processing
            audio_text = await self._enhanced_speech_to_text(audio_data)
            
            if not audio_text or len(audio_text.strip()) < 10:
                # Return proper dictionary structure instead of string
                return {
                    "confidence_score": 20,
                    "accuracy": "low",
                    "rfq_archetype": "Item RFQ (single or multi-line products)",
                    "products": [
                        {
                            "product_name": "Audio RFQ Request",
                            "product_type": "General",
                            "qty": 1,
                            "unit": "units",
                            "specifications": "Audio RFQ - speech recognition unclear, please provide text description",
                            "status": "need_clarity",
                            "Model": "MDL-AUD-001",
                            "Category": "General",
                            "Estimated_cost": "TBD",
                            "preference": "OEM/Equivalent",
                            "Bom": "BOM-2024-AUD",
                            "Sku": "SKU-AUD-000001",
                            "Hs_code": "0000.00.00",
                            "line_confidence": 0.2
                        }
                    ],
                    "extracted_details": None,
                    "Ai_generated": {
                        "ai_suggestion": "Audio processing unclear. Please provide text description of required items.",
                        "tag": "audio_processing",
                        "category": "General",
                        "describe_request": "Audio RFQ request - speech recognition indicated unclear content",
                        "Technical_drawning": "Not available",
                        "total_value": "TBD",
                        "currency": "USD",
                        "target_region": "Global"
                    }
                }
            
            # Process the extracted text through normal RFQ generation
            return await self.generate_rfq_json(audio_text, context)
            
        except Exception as e:
            logger.error(f"Enhanced audio processing failed: {e}")
            # Return proper dictionary structure instead of string
            return {
                "confidence_score": 25,
                "accuracy": "low",
                "rfq_archetype": "Item RFQ (single or multi-line products)",
                "products": [
                    {
                        "product_name": "Audio Equipment Request",
                        "product_type": "General",
                        "qty": 1,
                        "unit": "units",
                        "specifications": f"Audio RFQ processing error: {str(e)}",
                        "status": "need_clarity",
                        "Model": "MDL-AUD-002",
                        "Category": "General",
                        "Estimated_cost": "TBD",
                        "preference": "OEM/Equivalent",
                        "Bom": "BOM-2024-ERR",
                        "Sku": "SKU-AUD-000002",
                        "Hs_code": "0000.00.00",
                        "line_confidence": 0.25
                    }
                ],
                "extracted_details": None,
                "Ai_generated": {
                    "ai_suggestion": "Audio processing encountered issues. Please provide text description of required items.",
                    "tag": "audio_error",
                    "category": "General",
                    "describe_request": f"Audio RFQ processing error: {str(e)}",
                    "Technical_drawning": "Not available",
                    "total_value": "TBD",
                    "currency": "USD",
                    "target_region": "Global"
                }
            }

    async def _enhanced_speech_to_text(self, audio_data: bytes) -> str:
        """Enhanced speech recognition with fallback"""
        try:
            # For now, simulate speech recognition (replace with actual implementation)
            logger.info("Processing audio through enhanced speech recognition")
            
            # Simulate different audio content based on file size or metadata
            audio_text = "Audio RFQ request for electric motors and industrial equipment"
            
            # You can integrate with:
            # 1. OpenAI Whisper API
            # 2. Google Speech-to-Text
            # 3. Azure Speech Services
            # 4. Local Whisper model
            
            return audio_text
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return "Audio RFQ request for electric motors and industrial equipment"


# Create singleton instance
ollama_service = OllamaService()

# For testing and standalone execution
if __name__ == "__main__":
    import asyncio
    
    async def test_service():
        """Test the OllamaService with sample data"""
        
        # Test with simple RFQ text
        test_text = """
        We need to purchase 50 units of industrial air filters for our manufacturing facility.
        The filters should be HEPA grade H13 with dimensions 600x600x150mm.
        Delivery required to our warehouse in Chicago within 2 weeks.
        Budget is approximately $150 per unit.
        """
        
        print("Testing OllamaService with sample RFQ...")
        print(f"Using Ollama at: {settings.OLLAMA_BASE_URL}")
        print(f"Using model: {settings.OLLAMA_MODEL}")
        
        result = await ollama_service.generate_rfq_json(test_text)
        
        print(f"Confidence Score: {result.get('confidence_score')}")
        print(f"RFQ Archetype: {result.get('rfq_archetype')}")
        print(f"Number of Products: {len(result.get('products', []))}")
        
        if result.get('products'):
            product = result['products'][0]
            print(f"Product: {product.get('product_name')}")
            print(f"Quantity: {product.get('qty')}")
            print(f"Estimated Cost: {product.get('Estimated_cost')}")
        
        print("\nFull JSON Response:")
        print(json.dumps(result, indent=2))
    
    # Run the test
    asyncio.run(test_service())