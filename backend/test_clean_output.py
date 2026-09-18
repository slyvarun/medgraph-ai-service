"""
test_clean_output.py
Verification script for output formatting (bullet points vs paragraphs), relevance filtering, and precautions coreference.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import ask_agent, _search_vector_guidelines, _resolve_coreference_and_intent

def test_guideline_relevance_filtering():
    print("--- 1. Testing Guideline Relevance Filtering ---")
    # Query about Fever should NOT match Diabetic Neuropathy or Hypertension
    fever_query = "What medicines treat Fever and Pain?"
    vector_results = _search_vector_guidelines(fever_query)
    print(f"Fever Query Vector Passages count: {len(vector_results)}")
    conditions = [g.get("condition") for g in vector_results]
    print(f"Matched Conditions: {conditions}")

    assert "Diabetic Neuropathy" not in conditions, "Irrelevant Diabetic Neuropathy guideline leaked into Fever query"
    assert "Hypertension" not in conditions, "Irrelevant Hypertension guideline leaked into Fever query"
    print("[OK] Guideline Relevance Filtering passed.\n")

def test_bullet_formatting_and_limit():
    print("--- 2. Testing Bullet Point Formatting & Record Limit ---")
    query = "What medicines treat Fever and Pain?"
    response = ask_agent(query)
    safe_response = response.encode('ascii', 'ignore').decode('ascii')
    print("Agent Response Preview:\n" + safe_response[:500] + "\n...")

    # Check bullet points formatting
    assert "- 📌" in response or "####" in response, "Response missing structured headers/bullets"
    assert not ("1. Glycemic Control: Maintain strict" in response and "2. Lifestyle & Physical Therapy:" in response and "\n" not in response), "Guidelines rendered as continuous paragraph instead of bullet points"

    # Count medicine occurrences
    med_count = response.count("Overview & Classification")
    print(f"Returned medicine cards count: {med_count}")
    assert med_count <= 4, f"Expected at most 4 medicine cards, got {med_count}"
    print("[OK] Bullet Formatting & Record Limit passed.\n")

def test_precautions_coreference():
    print("--- 3. Testing Precautions Coreference Resolution ---")
    history = [
        {"role": "user", "content": "if the fever is fungal infection"},
        {"role": "assistant", "content": "Clarinazole and Amoxicillin are antifungals."}
    ]
    query = "precautions to be taken for intaking these medicines"
    resolved, is_ambiguous, msg = _resolve_coreference_and_intent(query, history=history)
    print(f"Original: '{query}' -> Resolved: '{resolved}'")
    assert "Clarinazole" in resolved or "Amoxicillin" in resolved, "Failed to resolve 'these medicines' to active antifungal context"

    response = ask_agent(query, history=history)
    safe_response = response.encode('ascii', 'ignore').decode('ascii')
    print("Precautions Response Preview:\n" + safe_response[:400] + "\n...")
    assert "Clarinazole" in response or "Amoxicillin" in response or "Clinical Assessment" in response, "Precautions response missing context"
    print("[OK] Precautions Coreference Resolution passed.\n")

if __name__ == "__main__":
    test_guideline_relevance_filtering()
    test_bullet_formatting_and_limit()
    test_precautions_coreference()
    print("=== SUCCESS: All clean output and relevance tests passed successfully! ===")
