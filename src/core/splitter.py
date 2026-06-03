"""
Document Splitter - Breaking Up Large Files

When you upload a document (PDF or text file), it gets broken into smaller chunks.
This makes searching more precise - instead of finding a whole 50-page document,
we can find the specific paragraph that answers your question.

The splitter tries to keep content together logically:
- Paragraph breaks
- Code blocks stay intact
- Lists stay together
- Overlap between chunks helps maintain context
"""

import logging
from typing import List

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger("RAG.Splitter")


class RecursiveCharacterSplitter:
    """
    Splits text by looking at separators in priority order:
    paragraphs → newlines → sentences → spaces → characters.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunks.append(text[start:])
                break

            # Try to find a good separator split point
            split_pos = -1
            for sep in self.separators:
                if sep == "":
                    split_pos = end
                    break
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos >= start + self.overlap:
                    split_pos = pos + len(sep)
                    break

            chunks.append(text[start:split_pos])
            start = split_pos - self.overlap
            if start < 0:
                start = split_pos

        return chunks
