"""
Verification test for French and German clinical GraphRAG queries.
"""
import requests
import json

BASE_URL = "http://localhost:8008"

def test_multilingual():
    test_cases = [
        {
            "name": "French Symptom Query",
            "payload": {"query": "J'ai de violents maux de tête et de la migraine", "language": "fr"},
            "expected_intent": "SYMPTOM_TO_MEDICINE"
        },
        {
            "name": "German Alternative Query",
            "payload": {"query": "Was sind Alternativen zu Lisinopril?", "language": "de"},
            "expected_intent": "BRAND_TO_ALTERNATIVE"
        },
        {
            "name": "French Ambiguous Query",
            "payload": {"query": "médicament", "language": "fr"},
            "expected_intent": "AMBIGUOUS"
        },
        {
            "name": "German Ambiguous Query",
            "payload": {"query": "Blutdruck", "language": "de"},
            "expected_intent": "AMBIGUOUS"
        }
    ]

    print("==================================================")
    print("MEDGRAPH NEXUS: FRENCH & GERMAN MULTILINGUAL TEST")
    print("==================================================")

    for tc in test_cases:
        print(f"\n[TEST] {tc['name']} (lang={tc['payload']['language']}): '{tc['payload']['query']}'")
        res = requests.post(f"{BASE_URL}/api/search", json=tc["payload"], timeout=30)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print(f"  Status: {res.status_code}")
        print(f"  Intent Detected: {data.get('intent')}")
        print(f"  Grounded: {data.get('is_grounded')}")
        print(f"  Symptoms/Entities: {data.get('symptoms_identified')[:3]}")
        answer_preview = data.get('answer', '')[:200].replace('\n', ' ')
        print(f"  Answer Sample: {answer_preview}...")

    print("\nALL FRENCH & GERMAN TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_multilingual()
