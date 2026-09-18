"""
test_conversational_rag.py
Verification script for conversational RAG, coreference resolution, intent parsing, and category guardrails.
"""

import sys
import os

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from query_agent import ask_agent, _search_local_cache, _resolve_coreference_and_intent

def test_intent_clarification():
    print("--- 1. Testing Intent & Ambiguity Clarification ---")
    ambiguous_queries = ["give me medicine", "drugs", "help", "tell me more"]
    for q in ambiguous_queries:
        resolved, is_ambiguous, msg = _resolve_coreference_and_intent(q)
        print(f"Query: '{q}' -> Ambiguous: {is_ambiguous}")
        assert is_ambiguous, f"Expected '{q}' to be flagged as ambiguous"
        assert "Could you please specify" in msg, "Expected clarification prompt"
    print("[OK] Intent Clarification passed.\n")

def test_coreference_resolution():
    print("--- 2. Testing Coreference Resolution ---")
    history = [
        {"role": "user", "content": "Why do we use Ibuprocillin?"},
        {"role": "assistant", "content": "Ibuprocillin is an Antiviral prescribed for Infection."}
    ]
    follow_up = "What is its dosage form?"
    resolved, is_ambiguous, msg = _resolve_coreference_and_intent(follow_up, history=history)
    print(f"Original: '{follow_up}' -> Resolved: '{resolved}'")
    assert "Ibuprocillin" in resolved, "Coreference resolution failed to resolve 'its' to 'Ibuprocillin'"

    # Execute full RAG agent with history
    response = ask_agent(follow_up, history=history)
    safe_response = response[:200].encode('ascii', 'ignore').decode('ascii')
    print(f"Agent response preview:\n{safe_response}...\n")
    assert "Ibuprocillin" in response or "Injection" in response, "RAG response missing resolved medicine context"
    print("[OK] Coreference Resolution passed.\n")

def test_category_guardrails():
    print("--- 3. Testing Category Guardrails ---")
    # Search local cache for Dextrophen (Category: Antibiotic)
    res = _search_local_cache("Dextrophen")
    assert len(res) > 0, "No records returned for Dextrophen"
    target = res[0]
    med = target["medicine"]
    alts = target["alternatives"]
    print(f"Medicine: {med['name']} | Category: {med['category']} | Indication: {med['indication']}")
    print(f"Filtered Alternatives: {alts}")

    # Verify that returned alternatives share the same category (or compatible)
    from query_agent import _get_local_cache
    records = _get_local_cache().get("records", [])
    med_by_name = {r["name"]: r for r in records}

    for alt_name in alts:
        alt_rec = med_by_name.get(alt_name)
        if alt_rec and med.get("category"):
            assert alt_rec.get("category") == med["category"], f"Category mismatch: {alt_name} ({alt_rec.get('category')}) vs {med['name']} ({med['category']})"

    print("[OK] Category Guardrails passed.\n")

if __name__ == "__main__":
    test_intent_clarification()
    test_coreference_resolution()
    test_category_guardrails()
    print("=== SUCCESS: All Conversational RAG & Guardrail tests passed successfully! ===")
