"""
Grounding Verification - Making Sure Answers Are Based on Real Facts

This service checks that the AI's answers actually match the documents we found.
It prevents "hallucinations" (making up facts) by comparing each sentence
of the answer against the source documents using semantic similarity.

If the answer isn't well-grounded, we can flag or fix it.
"""

import logging
import re
import threading
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.core.config import EMBEDDING_MODEL

logger = logging.getLogger("RAG.Services.Grounding")

_embedding_model_instance: Optional["SentenceTransformer"] = None
_embedding_model_lock = threading.Lock()


def _get_shared_embedding_model():
    """Get or load the shared embedding model (loaded once, reused everywhere)."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        with _embedding_model_lock:
            if _embedding_model_instance is None:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model for grounding checks: {EMBEDDING_MODEL}")
                _embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model_instance


class GroundingVerifier:
    """
    Checks that AI answers are actually based on the provided documents.
    
    This prevents the AI from making up facts by comparing each sentence
    against the source content. It uses semantic similarity to find matches
    even when the wording is different.
    """

    def __init__(self):
        self._embedding_model = None

    def _get_embedding_model(self):
        return _get_shared_embedding_model()

    def extract_citation_markers(self, answer: str) -> List[Dict]:
        """Find citations like [source: filename] in the answer text."""
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

    def _check_entity_hallucination(self, sentence: str, context_chunks: List[str]) -> float:
        """
        Returns a penalty multiplier between 0.0 and 1.0.
        1.0 means no hallucinated entities or numbers were found.
        Lower values mean entities or numbers were found in the sentence but not in context.
        """
        context_lower = " ".join(context_chunks).lower()
        
        # 1. Check numbers/digits (e.g., years, values) of length >= 2
        sent_numbers = re.findall(r'\b\d{2,}\b', sentence)
        for num in sent_numbers:
            if num not in context_lower:
                return 0.35  # Penalty for hallucinated numbers/years

        # 2. Check proper nouns / names (capitalized words)
        words = re.findall(r'\b[A-Z][a-zA-Z]*\b', sentence)
        if not words:
            return 1.0

        first_word = True
        for word in words:
            # Ignore common first words of sentences when checking
            if first_word:
                first_word = False
                if word.lower() in {
                    "the", "a", "an", "in", "on", "at", "this", "it", "they", "we", "he", "she", 
                    "there", "when", "where", "how", "why", "who", "then", "if", "for", "as", 
                    "by", "with", "but", "and", "or", "python"
                }:
                    continue

            # Check if the proper noun is in context (case-insensitive)
            if word.lower() not in context_lower:
                return 0.35  # Penalty for hallucinated proper noun/name

        return 1.0

    def verify_sentence_support(self, sentence: str, context_chunks: List[str], threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Check if a sentence is backed by any of the source documents.
        
        Uses semantic similarity to find if the sentence matches any document,
        even when the wording is different. Falls back to word overlap if
        the embedding model fails.
        
        Returns: (is_supported, best_matching_score)
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
            
            # Apply entity/number hallucination penalty
            penalty = self._check_entity_hallucination(sentence, context_chunks)
            best_score = best_score * penalty
            
            return best_score >= threshold, best_score
        except Exception:
            # Fallback: check for common words
            sentence_lower = sentence.lower()
            overlap = len(set(sentence_lower.split()) & set(" ".join(context_chunks).lower().split()))
            ratio = overlap / max(len(sentence_lower.split()), 1)
            
            # Apply penalty to fallback ratio as well
            penalty = self._check_entity_hallucination(sentence, context_chunks)
            ratio = ratio * penalty
            
            return ratio >= threshold, ratio

    def detect_hallucinations(self, answer: str, context_chunks: List[str]) -> List[Dict]:
        """
        Find sentences in the answer that don't match the source documents.
        
        Flags suspicious claims that might be made up, including:
        - Absolute claims (always, never, everyone)
        - Superlatives (best, worst, only)
        - Specific dates/years
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
        """Calculate how well the answer is grounded in source documents (0.0 to 1.0)."""
        score, _ = self.verify_grounding(answer, context_chunks)
        return score

    def verify_grounding(self, answer: str, context_chunks: List[str], threshold: float = 0.3) -> Tuple[float, List[str]]:
        """
        Check that each sentence in the answer matches the source documents.
        
        Returns: (grounding_score, list_of_unsupported_claims)
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+\s*', answer) if len(s.strip()) > 10]

        if not sentences:
            return 1.0, []

        supported_count = 0
        total_score = 0.0
        unsupported_claims = []

        for sent in sentences:
            is_supported, score = self.verify_sentence_support(sent, context_chunks, threshold=threshold)
            if is_supported:
                supported_count += 1
                total_score += score
            else:
                total_score += score
                unsupported_claims.append(sent[:150])

        # Calculate overall score: mix of support ratio and average similarity
        sentence_score = supported_count / len(sentences)
        avg_similarity = total_score / len(sentences)
        grounding_score = round((sentence_score * 0.6 + avg_similarity * 0.4), 3)

        return grounding_score, unsupported_claims


class GroundingEnforcer:
    """
    Adds citations to answers and warns about poorly-grounded responses.
    """

    def __init__(self, llm_client):
        self.verifier = GroundingVerifier()
        self.llm_client = llm_client

    def enforce_citations(self, answer: str, context: str) -> str:
        """
        Add [source: filename] citations to sentences that match document content.
        
        If the answer doesn't have citations, this adds them based on which
        documents each sentence seems to come from.
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
                # Check if sentence shares meaningful words with this source
                matched_words = [word for word in sent.lower().split() if len(word) > 4 and word in content.lower()]
                if matched_words:
                    sent = f'{sent} [source: {src}]'
                    break

            result.append(sent + punct)

        return "".join(result)

    def add_groundedness_warning(self, answer: str, context: str) -> str:
        """Prepend a warning if the answer doesn't seem well-grounded in documents."""
        grounding_score = self.verifier.compute_grounding_score(answer, context)

        if grounding_score < 0.5:
            warning = f"[WARNING: Low grounding score ({grounding_score:.2f}). Answer may contain unverified claims.]\n\n"
            return warning + answer

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