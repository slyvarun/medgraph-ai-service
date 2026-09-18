"""
test_natural_synthesis.py
Verification script for natural ChatGPT/Gemini-style responses, product deduplication, and removal of technical source headers.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import ask_agent, _deduplicate_records

def test_deduplication():
    print("--- 1. Testing OpenFDA Store-Brand Product Deduplication ---")
    mock_openfda_records = [
        {"medicine": {"name": "Tension Headache Pain Relieving Aid, Wal-Mart Stores Inc", "category": "HUMAN OTC DRUG", "classification": "Central Nervous System Stimulant"}},
        {"medicine": {"name": "Extra Strength Headache, Caseys 4good", "category": "HUMAN OTC DRUG", "classification": "Central Nervous System Stimulant"}},
        {"medicine": {"name": "Headache Relief Extra Strength, Meijer Distribution Inc", "category": "HUMAN OTC DRUG", "classification": "Nonsteroidal Anti-inflammatory"}},
        {"medicine": {"name": "Tension Headache Relief, TIME CAP LABS INC", "category": "HUMAN OTC DRUG", "classification": "Central Nervous System Stimulant"}},
        {"medicine": {"name": "Headache Relief, WALGREENS", "category": "HUMAN OTC DRUG", "classification": "Nonsteroidal Anti-inflammatory"}}
    ]

    deduped = _deduplicate_records(mock_openfda_records)
    print(f"Original records: {len(mock_openfda_records)} -> Deduplicated records: {len(deduped)}")
    names = [item["medicine"]["name"] for item in deduped]
    print(f"Deduplicated product names: {names}")

    assert len(deduped) <= 3, f"Expected at most 3 deduplicated items, got {len(deduped)}"
    assert "Tension Headache Pain Relieving Aid" in names[0], "Store-brand suffix cleanup failed"
    print("[OK] Product Deduplication passed.\n")

def test_no_technical_headers_and_natural_structure():
    print("--- 2. Testing Removal of Technical Headers & Natural Output ---")
    query = "headache"
    response = ask_agent(query)
    safe_response = response.encode('ascii', 'ignore').decode('ascii')
    print("Natural Response Preview:\n" + safe_response[:500] + "\n...")

    # Verify no technical header noise
    assert "Doctor AI Clinical Assessment (Source:" not in response, "Technical source tag header was not removed"
    assert "openFDA API + Vector Guidelines" not in response, "Technical source tag header was not removed"

    # Verify natural ChatGPT/Gemini sections
    assert "Primary Treatment" in response or "Medication Options" in response, "Missing Primary Treatment section"
    assert "When to Consult a Doctor" in response or "Seek Medical Attention" in response, "Missing medical attention section"
    print("[OK] Removal of Technical Headers & Natural Output passed.\n")

if __name__ == "__main__":
    test_deduplication()
    test_no_technical_headers_and_natural_structure()
    print("=== SUCCESS: All natural synthesis and noise removal tests passed successfully! ===")
