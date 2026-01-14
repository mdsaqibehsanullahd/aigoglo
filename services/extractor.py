"""
Multi-Format Extraction Layer
Handles: PDF, Excel, Word, Text, Audio, Image, JSON
"""
import io
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import PyPDF2
import openpyxl
import pandas as pd
import json
from PIL import Image, ImageEnhance  # Add ImageEnhance
import re  # Add for text processing

logger = logging.getLogger(__name__)


class ExtractorService:
    """
    Multi-format text extraction service
    Converts all input formats to raw text for LLM processing
    """
    
    async def extract_from_pdf(self, file_content: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF
        Returns: (extracted_text, metadata)
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            metadata = {
                "page_count": len(pdf_reader.pages),
                "format": "PDF"
            }
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{text}")
            
            extracted_text = "\n\n".join(text_parts)
            
            if not extracted_text.strip():
                logger.warning("No text extracted from PDF")
                extracted_text = "Empty PDF or image-based PDF (OCR required)"
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return f"PDF extraction failed: {str(e)}", {"format": "PDF", "error": str(e)}
    
    async def extract_from_excel(self, file_content: bytes, filename: str = "file.xlsx") -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from Excel/CSV
        Handles BOM/BOQ formats
        """
        try:
            # Determine if CSV or Excel
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content))
                metadata = {"format": "CSV", "rows": len(df), "columns": len(df.columns)}
            else:
                excel_file = io.BytesIO(file_content)
                workbook = openpyxl.load_workbook(excel_file, data_only=True)
                
                # Read all sheets
                all_sheets_data = []
                for sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    df = pd.DataFrame(sheet.values)
                    all_sheets_data.append(f"--- Sheet: {sheet_name} ---\n{df.to_string()}")
                
                extracted_text = "\n\n".join(all_sheets_data)
                metadata = {
                    "format": "Excel",
                    "sheets": workbook.sheetnames,
                    "sheet_count": len(workbook.sheetnames)
                }
                
                return extracted_text, metadata
            
            # For CSV
            extracted_text = df.to_string()
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"Excel extraction error: {e}")
            return f"Excel extraction failed: {str(e)}", {"format": "Excel", "error": str(e)}
    
    async def extract_from_image(self, file_content: bytes) -> Tuple[str, Dict[str, Any]]:
        """Enhanced image processing with pre-processing"""
        try:
            # Pre-process image for better OCR
            enhanced_image = self._preprocess_image(file_content)
            
            image = Image.open(io.BytesIO(enhanced_image))
            metadata = {
                "format": "Image",
                "size": image.size,
                "mode": image.mode,
                "enhanced": True
            }
            
            return enhanced_image, metadata
            
        except Exception as e:
            logger.error(f"Enhanced image processing error: {e}")
            # Fallback to original image
            return file_content, {"format": "Image", "error": str(e), "enhanced": False}

    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """Apply image enhancements for better OCR"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            sharpener = ImageEnhance.Sharpness(image)
            image = sharpener.enhance(1.3)
            
            # Resize if too small (minimum 300 DPI equivalent)
            width, height = image.size
            if width < 1000:
                scale_factor = 1000 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save enhanced image
            output = io.BytesIO()
            image.save(output, format='PNG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Image pre-processing failed: {e}")
            return image_bytes  # Return original if enhancement fails
    
    async def extract_from_audio(self, file_content: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Process audio for Whisper model
        Returns raw bytes for audio processing
        """
        try:
            metadata = {
                "format": "Audio",
                "size_bytes": len(file_content)
            }
            
            return file_content, metadata
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return file_content, {"format": "Audio", "error": str(e)}
    
    async def extract_from_json(self, file_content: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Parse JSON RFQ
        Converts JSON to text representation
        """
        try:
            json_data = json.loads(file_content.decode('utf-8'))
            
            # Convert JSON to readable text
            extracted_text = json.dumps(json_data, indent=2)
            
            metadata = {
                "format": "JSON",
                "keys": list(json_data.keys()) if isinstance(json_data, dict) else []
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return f"JSON parsing failed: {str(e)}", {"format": "JSON", "error": str(e)}
    
    async def extract_from_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process plain text RFQ
        """
        metadata = {
            "format": "Text",
            "length": len(text)
        }
        
        return text, metadata
    
    def detect_file_type(self, filename: str, content_type: Optional[str] = None) -> str:
        """
        Detect file type from filename and content type
        """
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith(('.xlsx', '.xls', '.csv')):
            return 'excel'
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            return 'image'
        elif filename_lower.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
            return 'audio'
        elif filename_lower.endswith('.json'):
            return 'json'
        elif filename_lower.endswith('.txt'):
            return 'text'
        else:
            # Fallback to content type
            if content_type:
                if 'pdf' in content_type:
                    return 'pdf'
                elif 'excel' in content_type or 'spreadsheet' in content_type:
                    return 'excel'
                elif 'image' in content_type:
                    return 'image'
                elif 'audio' in content_type:
                    return 'audio'
                elif 'json' in content_type:
                    return 'json'
            
            return 'text'


# Singleton instance
extractor_service = ExtractorService()
