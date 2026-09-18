"""
test_humanized_persona.py
Verification script for Food-Drug/DDI Guidelines & Empathetic Clinician Persona.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import (
    ask_agent,
    _search_vector_guidelines,
    _resolve_coreference_and_intent
)

def test_food_drug_interaction_retrieval():
    print("--- 1. Testing Food-Drug & DDI Guideline Retrieval ---")
    query = "Can I drink grapefruit juice or alcohol with my medicine?"
    guidelines = _search_vector_guidelines(query)
    print(f"Retrieved Guidelines count for '{query}': {len(guidelines)}")

    has_food_safety = any(
        "Grapefruit" in g.get("title", "") or
        "Dietary" in g.get("title", "") or
        any("Grapefruit" in b or "Alcohol" in b for b in g.get("bullets", []))
        for g in guidelines
    )
    assert has_food_safety, "Food-Drug safety protocol missing from retrieved guidelines"
    print("[OK] Food-Drug & DDI Guideline Retrieval passed.\n")

def test_empathetic_narrative_opening():
    print("--- 2. Testing Empathetic Narrative Openings ---")
    query = "When should I take Ibuprocillin?"
    response = ask_agent(query)
    safe_resp = response.encode('ascii', 'ignore').decode('ascii')
    print("Response Opening Preview:\n" + safe_resp[:300] + "\n")

    assert "Got it" in response or "Let's" in response or "Prescribed" in response, "Empathetic narrative opening missing"
    print("[OK] Empathetic Narrative Openings passed.\n")

def test_warm_ambiguity_recovery():
    print("--- 3. Testing Warm Ambiguity Recovery ---")
    vague_query = "help me"
    resolved, is_ambiguous, msg = _resolve_coreference_and_intent(vague_query, history=[])
    print(f"Ambiguity Response Preview:\n{msg[:250]}\n")

    assert is_ambiguous, "Vague query 'help me' should be flagged as ambiguous"
    assert "Hmm, I want to make sure I give you accurate clinical guidance" in msg or "somatic" in msg or "Could you please specify" in msg, "Warm clinician clarification text missing"
    print("[OK] Warm Ambiguity Recovery passed.\n")

if __name__ == "__main__":
    test_food_drug_interaction_retrieval()
    test_empathetic_narrative_opening()
    test_warm_ambiguity_recovery()
    print("=== SUCCESS: All Food-Drug & Empathetic Persona tests passed successfully! ===")
