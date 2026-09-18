"""
test_agentic_reasoning.py
Verification script for Multi-Step Agentic Reasoning Loop (ReAct / Plan-and-Solve).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import (
    ask_agent,
    _decompose_query,
    _multi_hop_graph_reasoning,
    _self_critique_and_correct,
    PatientClinicalState
)

def test_query_decomposition():
    print("--- 1. Testing Agentic Query Decomposition ---")
    query = "I have headache and vomits so what could i take"
    sub_queries = _decompose_query(query)
    print(f"Original Query: '{query}' -> Decomposed: {sub_queries}")

    assert any("headache" in sq.lower() for sq in sub_queries), "Failed to extract 'headache' sub-query"
    assert any("vomit" in sq.lower() for sq in sub_queries), "Failed to extract 'vomits' sub-query"
    print("[OK] Agentic Query Decomposition passed.\n")

def test_patient_state_allergy_memory():
    print("--- 2. Testing Patient Clinical State Memory ---")
    history = [
        {"role": "user", "content": "Amoxicillin makes me nauseous and I have an allergy to antibiotics."}
    ]
    patient_state = PatientClinicalState(history=history)
    print(f"Disqualified Drugs: {patient_state.disqualified_drugs}")
    print(f"Disqualified Categories: {patient_state.disqualified_categories}")

    assert "amoxicillin" in patient_state.disqualified_drugs or "antibiotic" in patient_state.disqualified_categories
    print("[OK] Patient Clinical State Memory passed.\n")

def test_self_critique_and_correct():
    print("--- 3. Testing Self-Correction & Reflection Loop ---")
    query = "I have headache and vomits so what could i take"
    patient_state = PatientClinicalState()
    
    mock_candidates = [
        {"medicine": {"name": "Acetocillin", "category": "antibiotic", "indication": "Infection"}},
        {"medicine": {"name": "Ibuprofen", "category": "analgesic", "indication": "Headache"}}
    ]
    mock_guidelines = [
        {"title": "Diabetic Neuropathy Care", "condition": "Diabetic Neuropathy"}
    ]

    corrected_c, corrected_g, mismatch_caught = _self_critique_and_correct(
        query, mock_candidates, mock_guidelines, patient_state
    )

    print(f"Self-Critique Caught Mismatch: {mismatch_caught}")
    print(f"Corrected Candidates Count: {len(corrected_c)}")
    print(f"Corrected Guidelines Count: {len(corrected_g)}")

    assert mismatch_caught, "Self-critique failed to catch antibiotic/neuropathy mismatch"
    assert len(corrected_c) == 1 and corrected_c[0]["medicine"]["name"] == "Ibuprofen"
    assert len(corrected_g) == 0
    print("[OK] Self-Correction & Reflection Loop passed.\n")

def test_end_to_end_agentic_reasoning():
    print("--- 4. Testing End-to-End Agentic Reasoning Loop ---")
    query = "I have headache and vomits so what could i take"
    response = ask_agent(query)
    safe_resp = response.encode('ascii', 'ignore').decode('ascii')
    print("Agentic Response Preview:\n" + safe_resp[:500] + "\n...")

    assert "Antibiotic" not in response and "Amoxicil" not in response, "Irrelevant antibiotic outputted for headache/vomits"
    assert "Disclaim" in response or "reference" in response
    print("[OK] End-to-End Agentic Reasoning Loop passed.\n")

if __name__ == "__main__":
    test_query_decomposition()
    test_patient_state_allergy_memory()
    test_self_critique_and_correct()
    test_end_to_end_agentic_reasoning()
    print("=== SUCCESS: All Multi-Step Agentic Reasoning Loop tests passed successfully! ===")
