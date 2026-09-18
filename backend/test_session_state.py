"""
test_session_state.py
Verification script for Active Conversational Session State Manager & Context Guard.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import (
    ask_agent,
    PatientClinicalState,
    _resolve_coreference_and_intent,
    search_openfda
)

def test_active_entity_extraction():
    print("--- 1. Testing Active Entity Extraction ---")
    history = [
        {"role": "user", "content": "When should I take Ibuprocillin?"},
        {"role": "assistant", "content": "Ibuprocillin is an antiviral prescribed for Infection."}
    ]
    state = PatientClinicalState(history=history)
    print(f"Extracted Active Entity: '{state.active_entity}'")
    assert state.active_entity == "Ibuprocillin", f"Expected 'Ibuprocillin', got '{state.active_entity}'"
    print("[OK] Active Entity Extraction passed.\n")

def test_context_injection_guard():
    print("--- 2. Testing Context Injection Guard ---")
    history = [
        {"role": "user", "content": "When should I take Ibuprocillin?"},
        {"role": "assistant", "content": "Ibuprocillin is an antiviral prescribed for Infection."}
    ]
    vague_prompt = "how to use this"
    resolved, is_ambiguous, msg = _resolve_coreference_and_intent(vague_prompt, history=history)
    print(f"Original Prompt: '{vague_prompt}' -> Guard Resolved: '{resolved}'")

    assert "Ibuprocillin" in resolved, "Context Injection Guard failed to inject 'Ibuprocillin'"
    assert not is_ambiguous, "Vague prompt should have been resolved via Context Injection Guard"
    print("[OK] Context Injection Guard passed.\n")

def test_openfda_stopword_guard():
    print("--- 3. Testing openFDA Stopword Guard ---")
    vague_query = "how to use this"
    results = search_openfda(vague_query)
    print(f"openFDA Results count for '{vague_query}': {len(results)}")
    assert len(results) == 0, "openFDA Stopword Guard failed to suppress raw pronoun query"
    print("[OK] openFDA Stopword Guard passed.\n")

def test_end_to_end_vague_followup():
    print("--- 4. Testing End-to-End Vague Follow-Up ---")
    history = [
        {"role": "user", "content": "When should I take Ibuprocillin?"},
        {"role": "assistant", "content": "Ibuprocillin is an antiviral prescribed for Infection."}
    ]
    vague_prompt = "how to use this"
    response = ask_agent(vague_prompt, history=history)
    safe_resp = response.encode('ascii', 'ignore').decode('ascii')
    print("End-to-End Response Preview:\n" + safe_resp + "\n")

    assert "Ibuprocillin" in response, "Response missing active entity 'Ibuprocillin'"
    assert "Sunscreen" not in response and "Glow This Way" not in response, "Response contains hallucinated openFDA sunscreen product"
    assert "Dosage Form" in response or "Injection" in response or "Prescribed" in response, "Response missing usage/dosage details"
    print("[OK] End-to-End Vague Follow-Up passed.\n")

if __name__ == "__main__":
    test_active_entity_extraction()
    test_context_injection_guard()
    test_openfda_stopword_guard()
    test_end_to_end_vague_followup()
    print("=== SUCCESS: All Conversational Session State Manager tests passed successfully! ===")
