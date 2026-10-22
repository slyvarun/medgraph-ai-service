import requests
import json
import time

BASE_URL = "http://localhost:8008/ask"

TEST_CASES = [
    {
        "path": "Path A (Symptom -> Medicine)",
        "query": "my head is pounding and throbbing",
        "expected_intent": "SYMPTOM_TO_MEDICINE"
    },
    {
        "path": "Path B (Brand -> Alternative)",
        "query": "What are alternatives to Lisinopril?",
        "expected_intent": "BRAND_TO_ALTERNATIVE"
    },
    {
        "path": "Path C (Ambiguous / Triage)",
        "query": "blood pressure",
        "expected_intent": "AMBIGUOUS"
    }
]

print("=" * 80)
print("MEDGRAPH NEXUS - DUAL-PATH INTENT ROUTING VERIFICATION SUITE")
print("=" * 80)

for idx, tc in enumerate(TEST_CASES, 1):
    print(f"\n[TEST {idx}/3] {tc['path']}")
    print(f"Query: \"{tc['query']}\"")
    t0 = time.time()
    try:
        resp = requests.post(BASE_URL, json={"query": tc["query"]}, timeout=30)
        elapsed = round(time.time() - t0, 2)
        if resp.status_code == 200:
            data = resp.json()
            intent = data.get("intent")
            grounded = data.get("is_grounded")
            answer = data.get("answer", "")
            print(f"Status: HTTP 200 ({elapsed}s)")
            print(f"Classified Intent: {intent}")
            print(f"Is Grounded: {grounded}")
            print(f"Sample Answer Preview:")
            print("-" * 60)
            print(answer[:500] + ("..." if len(answer) > 500 else ""))
            print("-" * 60)
        else:
            print(f"[FAIL] HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")

print("\n" + "=" * 80)
print("TEST SUITE COMPLETED")
print("=" * 80)
