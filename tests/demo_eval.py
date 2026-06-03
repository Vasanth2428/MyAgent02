"""
Demo script to show evaluation framework output
"""

import sys
sys.path.insert(0, '.')

from src.core.evaluator import GroundingVerifier

print('=== GROUNDING VERIFIER DEMO ===')
print()

# Test 1: Well-grounded answer
answer = 'Paris is the capital of France.'
context = 'Paris is the capital of France. It has the Eiffel Tower.'
score = GroundingVerifier.compute_grounding_score(answer, context)
print(f'Test 1 - Well-grounded answer:')
print(f'  Answer: {answer}')
print(f'  Grounding Score: {score}')
print(f'  Status: {"OK - Pass" if score > 0.7 else "FAIL"}')
print()

# Test 2: Hallucinated answer  
answer = 'The Eiffel Tower was built in 1889 and is 330 meters tall.'
context = 'Paris is the capital of France.'
score = GroundingVerifier.compute_grounding_score(answer, context)
print(f'Test 2 - Hallucinated answer:')
print(f'  Answer: {answer}')
print(f'  Grounding Score: {score}')
print(f'  Status: {"OK - Detected" if score < 0.5 else "FAIL"}')
print()

# Test 3: Citation extraction
answer = 'The password is SuperSecretAgent123 [source: config.txt]'
citations = GroundingVerifier.extract_citation_markers(answer)
print(f'Test 3 - Citation extraction:')
print(f'  Answer: {answer}')
print(f'  Citations found: {len(citations)}')
print(f'  Status: {"OK - Extracted" if len(citations) > 0 else "FAIL"}')