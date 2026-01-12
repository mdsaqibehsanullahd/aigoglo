"""
Normalization and Validation Service
Ensures output conforms to canonical schema
"""
import logging
from typing import Dict, Any
from models import RFQOutput
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class NormalizerService:
    """
    Validates and normalizes LLM output to canonical schema
    """
    
    def normalize_and_validate(self, llm_output: Dict[str, Any]) -> RFQOutput:
        """
        Validate LLM output against Pydantic schema
        Apply normalization rules
        """
        try:
            # Pydantic will validate and raise errors if schema violated
            rfq_output = RFQOutput(**llm_output)
            
            # Additional business rule validations
            self._validate_business_rules(rfq_output)
            
            return rfq_output
            
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            # Return fallback with error details
            return self._create_error_response(llm_output, str(e))
    
    def _validate_business_rules(self, rfq: RFQOutput):
        """
        Enforce business rules beyond schema validation
        """
        # Rule 1: At least one product must exist
        if not rfq.products or len(rfq.products) == 0:
            raise ValueError("At least one product is required")
        
        # Rule 2: Confidence score consistency
        if rfq.confidence_score < 30:
            logger.warning("Very low confidence score detected")
        
        # Rule 3: All products must have required fields (Pydantic handles this)
        for idx, product in enumerate(rfq.products):
            if not product.product_name or product.product_name.strip() == "":
                raise ValueError(f"Product {idx} has empty product_name")
    
    def _create_error_response(self, original_data: Dict[str, Any], error_msg: str) -> RFQOutput:
        """
        Create a valid fallback response when validation fails
        """
        from models import (
            Product, ExtractedDetails, AIGenerated, 
            RFQArchetype, ProductStatus, Preference
        )
        
        return RFQOutput(
            confidence_score=20,
            accuracy="low",
            rfq_archetype=RFQArchetype.ITEM_RFQ,
            products=[
                Product(
                    product_name="Validation Error - Manual Review Required",
                    product_type="Error",
                    qty=1,
                    unit="units",
                    specifications=f"Original data validation failed: {error_msg}",
                    status=ProductStatus.NEED_CLARITY,
                    Model="MDL-ERR-00001",
                    Category="Error",
                    Estimated_cost="N/A",
                    preference=Preference.OEM_OR_EQUIVALENT,
                    Bom="BOM-ERR-00001",
                    Sku="SKU-ERR-000001",
                    Hs_code="0000.00.00",
                    line_confidence=0.2
                )
            ],
            extracted_details=None,
            Ai_generated=AIGenerated(
                ai_suggestion="System validation error. Manual review required.",
                tag="validation_error",
                category="Error",
                describe_request="RFQ processing encountered validation error",
                Technical_drawning="Not available",
                total_value="N/A",
                currency="USD",
                target_region="Global"
            )
        )


# Singleton instance
normalizer_service = NormalizerService()
