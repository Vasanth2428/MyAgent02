"""
Document Splitter - Breaking Up Large Files

Thin adapter over LangChain's RecursiveCharacterTextSplitter.
The public interface (.split_text) is unchanged so all callers work without modification.

LangChain's implementation handles paragraph/sentence/word/character boundary splitting
natively with optimised routines, and supports token-based length functions (e.g. tiktoken).
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger("RAG.Splitter")


class RecursiveCharacterSplitter:
    """
    Adapter around LangChain's RecursiveCharacterTextSplitter.

    Splits text by looking at separators in priority order:
    paragraphs → newlines → sentences → spaces → characters.

    Preserves the original .split_text(text) -> List[str] interface.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.debug(
            f"RecursiveCharacterSplitter initialised (chunk_size={chunk_size}, overlap={overlap})"
        )

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self._splitter.split_text(text)
