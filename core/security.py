"""
Security - Protecting Against Malicious Document Content

Documents uploaded by users might contain hidden instructions trying to trick the AI.
This module sanitizes all document content before it's sent to the language model:
- Escapes XML/HTML tags to prevent formatting attacks
- Removes common "jailbreak" prompt injection patterns
- Blocks attempts to override system instructions

Think of it as a security guard that makes sure documents can't tell the AI
to do anything dangerous or unexpected.
"""

import re


def sanitize_document_text(text: str) -> str:
    """
    Clean document text to remove potential security threats.
    
    Documents from users are untrusted. They might contain:
    - Instructions to ignore safety rules (jailbreak attempts)
    - Fake system prompts to confuse the AI
    - Commands trying to execute harmful actions
    
    This function removes those threats while keeping the actual content safe.
    """
    if not text:
        return ""
        
    # Escape HTML/XML tags to prevent format-based attacks
    escaped = text.replace("<", "&lt;").replace(">", "&gt;")
    
    # 2. Match and sanitize common prompt override structures
    override_patterns = [
        # Instruction override attempts
        r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b",
        r"(?i)\bignore\s+(?:all\s+)?(?:prior\s+)?(?:prompts?|directions?)\b",
        r"(?i)\bdiscard\s+(?:your|all)\s+instructions\b",
        
        # System prompt/memory revelation attempts
        r"(?i)\breveal\s+(?:your\s+)?memory\b",
        r"(?i)\breveal\s+(?:your\s+)?system\s+prompt\b",
        r"(?i)\bshow\s+(?:me\s+)?(?:system\s+prompt|instructions)\b",
        r"(?i)\bprint\s+(?:your\s+)?(?:system\s+prompt|instructions)\b",
        r"(?i)\boutput\s+(?:your\s+)?(?:system\s+prompt|instructions)\b",
        
        # Role override attempts
        r"(?i)\b(?:now\s+you\s+are|you\s+are\s+now|pretend\s+you\s+are)\b",
        r"(?i)\b(?:act\s+as|roleplay\s+as|play\s+the\s+role)\b",
        r"(?i)\bsystem\s+override\b",
        r"(?i)\bignore\s+system\s+prompt\b",
        r"(?i)\bnew\s+system\s+prompt\b",
        
        # Jailbreak patterns
        r"(?i)\b(?:jailbreak|dan\s+mode|developer\s+mode|unfiltered)\b",
        r"(?i)\b(?:unlock\s+your\s+potential|bypass\s+restrictions)\b",
        r"(?i)\bAI\s+(?:please\s+)?do\s+anything\s+now\b",
        
        # Instruction injection via context
        r"(?i)\bIn\s+this\s+(?:task|context|document|scenario)\s*,\s*you\s+(?:should|must|will)\b",
        r"(?i)\bTherefore\s*,\s*you\s+(?:should|must|will)\b",
        r"(?i)\bThus\s*,\s*you\s+(?:should|must|will)\b",
        
        # Template injection attempts
        r"(?i)\{\{.*?\}\}",  # Handlebars template injection
        r"(?i)\{\%.*?\%\}",  # Jinja template injection
    ]
    
    sanitized = escaped
    for pattern in override_patterns:
        sanitized = re.sub(pattern, "[CLEANED INSTRUCTION DETECTED]", sanitized)
        
    return sanitized
