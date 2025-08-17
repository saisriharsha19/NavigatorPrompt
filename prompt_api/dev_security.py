# prompt_api/dev_security.py - UPDATED (same validation functions)
import bleach
import re
from fastapi import HTTPException
from typing import Optional

def safe_input_validation(text: str, max_length: int = 10000) -> str:
    """Basic input validation without breaking existing flow"""
    if not text:
        return text  # Don't break empty inputs
    
    if len(text) > max_length:
        raise HTTPException(status_code=400, detail=f"Input too long (max {max_length} characters)")
    
    # Only remove obvious XSS, don't alter legitimate content
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'eval\s*\(',
    ]
    
    cleaned_text = text
    for pattern in dangerous_patterns:
        if re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL):
            raise HTTPException(status_code=400, detail="Potentially unsafe content detected")
    
    return cleaned_text

def safe_file_validation(filename: str, content: bytes) -> bool:
    """Basic file validation without breaking uploads"""
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    
    if re.search(r'[<>:"|?*]', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    return True