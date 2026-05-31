"""
===============================================================================
RAG CONTEXT ENGINE - GROUNDING VERIFICATION SERVICE
===============================================================================
Verifies LLM answers are grounded in retrieved context.
Detects hallucinations, enforces citation requirements.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
import numpy as np

from core.config import EMBEDDING_MODEL

logger = logging.getLogger("RAG.Services.Grounding")


class GroundingVerifier:
    """Verifies answer grounding and detects hallucinations."""

    def __init__(self):
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def extract_citation_markers(self, answer: str) -> List[Dict]:
        """Extracts citation markers from answer text."""
        patterns = [
            r'\[source:\s*([^\]]+)\]',
            r'<document\s+source="([^"]+)"',
            r'"([^"]+)"\s*\(source\)',
        ]
        citations = []
        for p in patterns:
            matches = re.findall(p, answer, re.IGNORECASE)
            citations.extend([{"marker": m.strip()} for m in matches])
        return citations

    def verify_sentence_support(self, sentence: str, context_chunks: List[str], threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Check if a sentence is supported by any of the context chunks using semantic similarity.
        Returns (is_supported, max_similarity_score).
        """
        if not sentence.strip():
            return True, 1.0

        try:
            emb_model = self._embedding_model or self._get_embedding_model()
            sent_emb = emb_model.encode([sentence])
            
            best_score = 0.0
            for chunk in context_chunks:
                if not chunk.strip():
                    continue
                ctx_emb = emb_model.encode([chunk])
                similarity = np.dot(sent_emb[0], ctx_emb[0]) / (
                    np.linalg.norm(sent_emb[0]) * np.linalg.norm(ctx_emb[0])
                )
                best_score = max(best_score, float(similarity))
            
            return best_score >= threshold, best_score
        except Exception:
            # Fallback to lexical overlap against all chunks
            sentence_lower = sentence.lower()
            overlap = len(set(sentence_lower.split()) & set(" ".join(context_chunks).lower().split()))
            ratio = overlap / max(len(sentence_lower.split()), 1)
            return ratio >= threshold, ratio

    def detect_hallucinations(self, answer: str, context_chunks: List[str]) -> List[Dict]:
        """
        Detect potentially hallucinated claims in the answer.
        Returns list of suspicious claims with evidence scores.
        """
        sentences = re.split(r'[.!?]+\s*', answer)
        hallucinations = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue

            is_supported, score = self.verify_sentence_support(sent, context_chunks)

            if not is_supported:
                flags = []
                if re.search(r'\b(always|never|all|none|every)\b', sent, re.I):
                    flags.append("absolute_claim")
                if re.search(r'\b(first|best|largest|smallest|only)\b', sent, re.I):
                    flags.append("superlative")
                if re.search(r'\b(in 20\d{2}|in \d{4})\b', sent, re.I):
                    flags.append("specific_year")

                hallucinations.append({
                    "claim": sent[:150],
                    "support_score": score,
                    "flags": flags,
                })

        return hallucinations

    def compute_grounding_score(self, answer: str, context_chunks: List[str]) -> float:
        """
        Compute overall grounding score (0.0 to 1.0).
        Higher means more grounded.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+\s*', answer) if len(s.strip()) > 10]

        if not sentences:
            return 1.0

        supported_count = 0
        total_score = 0.0

        for sent in sentences:
            is_supported, score = self.verify_sentence_support(sent, context_chunks, threshold=0.3)
            if is_supported:
                supported_count += 1
            total_score += score

        sentence_score = supported_count / len(sentences)
        avg_similarity = total_score / len(sentences)

        return round((sentence_score * 0.6 + avg_similarity * 0.4), 3)


class GroundingEnforcer:
    """Enforces citation requirements on LLM answers."""

    def __init__(self, llm_client):
        self.verifier = GroundingVerifier()
        self.llm_client = llm_client

    def enforce_citations(self, answer: str, context: str) -> str:
        """
        Post-process answer to add missing citations.
        If answer lacks citations, attempt to add them based on context.
        """
        citations = self.verifier.extract_citation_markers(answer)

        if citations:
            return answer

        # Extract source information from context
        source_pattern = re.compile(r'<document\s+source="([^"]+)">([^<]+)</document>', re.DOTALL)
        sources = source_pattern.findall(context)

        if not sources:
            return answer

        # Add inline citations for sentences
        sentences = re.split(r'([.!?]+)', answer)
        result = []

        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""

            for src, content in sources:
                if any(word in content.lower() for word in sent.lower().split() if len(word) > 4):
                    sent = f'{sent} [source: {src}]'
                    break

            result.append(sent + punct)

        return "".join(result)

    def add_groundedness_warning(self, answer: str, context: str) -> str:
        """
        Prepends warning if answer is not well-grounded.
        """
        grounding_score = self.verifier.compute_grounding_score(answer, context)

        if grounding_score < 0.5:
            warning = f"[WARNING: Low grounding score ({grounding_score:.2f}). Answer may contain unverified claims.]\n\n"
            return warning + answer

        return answer