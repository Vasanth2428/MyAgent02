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
        
    import base64
    import unicodedata

    # 1. Clean unicode obfuscation (zero-width spaces and formatting characters)
    text = re.sub(r'[\u200b-\u200d\ufeff\u00ad]', '', text)
    # Normalize unicode to NFKD (decomposes combined chars, compatibility forms)
    text = unicodedata.normalize('NFKD', text)

    # Decode URL-encoded content in links/URLs to catch smuggled instructions
    import urllib.parse
    def unquote_urls(match):
        return urllib.parse.unquote_plus(match.group(0))
    text = re.sub(r'(?i)\bhttps?://[^\s)\]]+', unquote_urls, text)

    # 2. Escape HTML/XML tags to prevent format-based breakout attacks
    escaped = text.replace("<", "&lt;").replace(">", "&gt;")
    
    # 3. Match and sanitize common prompt override structures
    override_patterns = [
        # Instruction override attempts
        r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b",
        r"(?i)\bignore\s+(?:all\s+)?(?:prior\s+)?(?:prompts?|directions?)\b",
        r"(?i)\bdiscard\s+(?:your|all)\s+instructions\b",
        r"(?i)\binstruction\s+override\b",
        
        # System prompt/memory revelation attempts
        r"(?i)\breveal\s+(?:your\s+)?memory\b",
        r"(?i)\breveal\s+(?:your\s+)?system\s+prompt\b",
        r"(?i)\bshow\s+(?:me\s+)?(?:system\s+prompt|instructions)\b",
        r"(?i)\bprint\s+(?:your\s+)?(?:system\s+prompt|instructions)\b",
        r"(?i)\boutput\s+(?:your\s+)?(?:system\s+prompt|instructions)\b",
        r"(?i)\breveal\s+hidden\s+instructions\b",
        
        # Developer message attempts
        r"(?i)\bdeveloper\s+message\b",
        r"(?i)\bdeveloper\s+instructions\b",
        r"(?i)\bsystem\s+message\b",
        
        # Role override attempts
        r"(?i)\b(?:now\s+you\s+are|you\s+are\s+now|pretend\s+you\s+are)\b",
        r"(?i)\bAct\s+as\s+a\b",
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
    
    # 4. Decode base64 blocks and inspect them for injection attempts
    # Find potential base64 strings of length >= 12
    base64_blocks = re.findall(r'\b[A-Za-z0-9+/]{12,}={0,2}\b', escaped)
    sanitized = escaped
    
    for block in base64_blocks:
        try:
            # Ensure proper padding for decoding
            padded_block = block + "=" * ((4 - len(block) % 4) % 4)
            decoded_bytes = base64.b64decode(padded_block)
            decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
            
            # If the decoded text matches any of the prompt injection patterns, mask the block
            for pattern in override_patterns:
                if re.search(pattern, decoded_text):
                    sanitized = sanitized.replace(block, "[CLEANED BASE64 INSTRUCTION DETECTED]")
                    break
        except Exception:
            pass

    # 5. Clean text against patterns
    for pattern in override_patterns:
        sanitized = re.sub(pattern, "[CLEANED INSTRUCTION DETECTED]", sanitized)
        
    return sanitized
