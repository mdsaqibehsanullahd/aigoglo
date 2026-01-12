"""
Pydantic Models - Single Source of Truth for RFQ Schema
This is the CANONICAL OUTPUT CONTRACT
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from enum import Enum


class RFQArchetype(str, Enum):
    """RFQ Classification Types"""
    ITEM_RFQ = "Item RFQ (single or multi-line products)"
    PART_RFQ = "Part RFQ (Part No / Model / OEM spares)"
    SPEC_RFQ = "Spec RFQ (engineering specs, standards, drawings)"
    BOM_BOQ_RFQ = "BOM/BOQ RFQ (Excel list of items for projects)"
    SYSTEM_PACKAGE_RFQ = "System/Package RFQ (complete skid/system/line)"
    SERVICE_RFQ = "Service RFQ (installation, maintenance, EPC, inspection)"
    USED_SURPLUS_RFQ = "Used/Surplus RFQ (condition-based)"
    LOGISTICS_RFQ = "Logistics RFQ (shipping only)"
    HYBRID_RFQ = "Hybrid RFQ (mixture of any of the above)"


class ProductStatus(str, Enum):
    """Product line item status"""
    READY = "Ready"
    NEED_CLARITY = "need_clarity"
    INCOMPLETE = "incomplete"


class Preference(str, Enum):
    """OEM vs Equivalent preference"""
    OEM = "OEM"
    EQUIVALENT = "Equivalent"
    OEM_OR_EQUIVALENT = "OEM/Equivalent"


class Product(BaseModel):
    """
    Individual product line item
    ALL fields must be filled by LLM (no nulls allowed)
    """
    product_name: str = Field(..., description="Product name (LLM-inferred if needed)")
    product_type: str = Field(..., description="Product category/type")
    qty: int = Field(default=1, description="Quantity (LLM must infer if missing)")
    unit: str = Field(default="units", description="Unit of measurement")
    specifications: str = Field(..., description="Technical specifications (LLM-inferred)")
    status: ProductStatus = Field(default=ProductStatus.READY, description="Line item status")
    Model: str = Field(..., description="Model number (LLM-generated if not present)")
    Category: str = Field(..., description="Product category")
    Estimated_cost: str = Field(..., description="Estimated cost per unit (LLM-inferred)")
    preference: Preference = Field(default=Preference.OEM_OR_EQUIVALENT, description="OEM/Equivalent preference")
    Bom: str = Field(..., description="BOM reference (LLM-generated)")
    Sku: str = Field(..., description="SKU (LLM-generated)")
    Hs_code: str = Field(default="0000.00.00", description="HS Code (LLM-inferred)")
    line_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Per-line confidence score")


class ExtractedDetails(BaseModel):
    """
    Commercial/Logistics details
    ONLY section allowed to have null values
    """
    delivery_location: Optional[str] = Field(None, description="Delivery location")
    shipping_information: Optional[str] = Field(None, description="Shipping details")
    payment_term: Optional[str] = Field(None, description="Payment terms")
    inco_terms: Optional[str] = Field(None, description="Incoterms")
    submission_deadlines: Optional[str] = Field(None, description="Deadline for submission")
    target_budget: Optional[str] = Field(None, description="Budget range")


class AIGenerated(BaseModel):
    """
    AI-generated intelligence layer
    ALL fields must be filled by LLM
    """
    ai_suggestion: str = Field(..., description="AI-generated procurement suggestions")
    tag: str = Field(..., description="Classification tag")
    category: str = Field(..., description="Primary category")
    describe_request: str = Field(..., description="Natural language description of RFQ")
    Technical_drawning: str = Field(default="Not available", description="Drawing availability status")
    total_value: str = Field(..., description="Total estimated value")
    currency: str = Field(default="USD", description="Currency")
    target_region: str = Field(default="Global", description="Target region")


class RFQOutput(BaseModel):
    """
    CANONICAL RFQ OUTPUT SCHEMA
    This is the single source of truth
    """
    confidence_score: int = Field(..., ge=0, le=100, description="Overall RFQ confidence (0-100)")
    accuracy: Literal["low", "medium", "high"] = Field(..., description="Accuracy assessment")
    rfq_archetype: RFQArchetype = Field(..., description="RFQ classification type")
    products: List[Product] = Field(..., min_length=1, description="Product line items (minimum 1)")
    extracted_details: Optional[ExtractedDetails] = Field(None, description="Commercial details (nullable)")
    Ai_generated: AIGenerated = Field(..., description="AI intelligence layer")

    @field_validator('rfq_archetype', mode='before')
    @classmethod
    def normalize_archetype(cls, v):
        """Normalize archetype input to match enum values"""
        if isinstance(v, RFQArchetype):
            return v
        
        # Mapping of partial strings to full enum values
        archetype_map = {
            "Item RFQ": RFQArchetype.ITEM_RFQ,
            "Part RFQ": RFQArchetype.PART_RFQ,
            "Spec RFQ": RFQArchetype.SPEC_RFQ,
            "BOM/BOQ RFQ": RFQArchetype.BOM_BOQ_RFQ,
            "System/Package RFQ": RFQArchetype.SYSTEM_PACKAGE_RFQ,
            "Service RFQ": RFQArchetype.SERVICE_RFQ,
            "Used/Surplus RFQ": RFQArchetype.USED_SURPLUS_RFQ,
            "Logistics RFQ": RFQArchetype.LOGISTICS_RFQ,
            "Hybrid RFQ": RFQArchetype.HYBRID_RFQ,
        }
        
        # Try exact match first
        if v in [e.value for e in RFQArchetype]:
            return v
        
        # Try partial match
        for key, enum_val in archetype_map.items():
            if v.startswith(key):
                return enum_val.value
        
        # Default to Item RFQ if no match
        return RFQArchetype.ITEM_RFQ.value
    
    @field_validator('products')
    @classmethod
    def validate_products(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one product must be present")
        return v

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Confidence score must be between 0 and 100")
        return v


class RFQRequest(BaseModel):
    """Request model for text-based RFQ"""
    text: str = Field(..., description="Raw RFQ text input")
    context: Optional[str] = Field(None, description="Additional context")
