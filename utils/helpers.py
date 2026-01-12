"""
Utility Helper Functions
"""
import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict


def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"RFQ-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"


def generate_sku(category: str, sequence: int) -> str:
    """Generate SKU"""
    category_code = category[:3].upper()
    return f"SKU-{category_code}-{sequence:06d}"


def generate_bom(year: int, sequence: int) -> str:
    """Generate BOM reference"""
    return f"BOM-{year}-{sequence:04d}"


def generate_model(category: str, sequence: int) -> str:
    """Generate Model number"""
    category_code = category[:3].upper()
    return f"MDL-{category_code}-{sequence:05d}"


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency value"""
    return f"{amount:,.2f} {currency}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove any non-alphanumeric characters except dots, dashes, underscores
    safe_name = re.sub(r'[^\w\s\-\.]', '', filename)
    return safe_name.strip()
