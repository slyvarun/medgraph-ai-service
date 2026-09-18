"""
test_local_server.py
Tests the locally running MedGraph Nexus FastAPI server at http://localhost:8000.
"""

import httpx
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    with httpx.Client(timeout=10.0) as client:
        # 1. Health check
        r_health = client.get(f"{BASE_URL}/health")
        print(f"1. Health Check status: {r_health.status_code}")
        assert r_health.status_code == 200, "Health check failed"
        print(f"   Payload: {r_health.json()}\n")

        # 2. Graph stats
        r_stats = client.get(f"{BASE_URL}/graph-stats")
        print(f"2. Graph Stats status: {r_stats.status_code}")
        assert r_stats.status_code == 200, "Stats check failed"
        print(f"   Metadata: {r_stats.json().get('metadata')}\n")

        # 3. Test Ambiguity Clarification on /ask
        r_vague = client.post(f"{BASE_URL}/ask", json={"question": "give me medicine", "language": "en"})
        print(f"3. Vague Query /ask status: {r_vague.status_code}")
        assert r_vague.status_code == 200
        print(f"   Clarification response preview: {r_vague.json().get('answer')[:120]}...\n")

        # 4. Test Hybrid Search RRF on /ask
        q = "What are the best long-term management strategies for diabetic neuropathy alongside medication?"
        r_hybrid = client.post(f"{BASE_URL}/ask", json={"question": q, "language": "en"})
        print(f"4. Hybrid Search /ask status: {r_hybrid.status_code}")
        assert r_hybrid.status_code == 200
        ans = r_hybrid.json().get('answer', '')
        safe_ans = ans.encode('ascii', 'ignore').decode('ascii')
        print(f"   Hybrid Answer preview: {safe_ans[:250]}...\n")

        # 5. Test Conversational State with history
        history = [
            {"role": "user", "content": "Tell me about Ibuprocillin"},
            {"role": "assistant", "content": "Ibuprocillin is an antiviral prescribed for Infection."}
        ]
        r_conv = client.post(f"{BASE_URL}/ask", json={"question": "What is its dosage form?", "history": history, "language": "en"})
        print(f"5. Conversational Follow-up status: {r_conv.status_code}")
        assert r_conv.status_code == 200
        safe_conv = r_conv.json().get('answer', '').encode('ascii', 'ignore').decode('ascii')
        print(f"   Conversational Answer preview: {safe_conv[:250]}...\n")

        # 6. Test Subgraph Visualizer
        r_graph = client.get(f"{BASE_URL}/graph-subgraph", params={"q": "Ibuprocillin"})
        print(f"6. Subgraph Visualizer status: {r_graph.status_code}")
        assert r_graph.status_code == 200
        print(f"   Nodes count: {len(r_graph.json().get('graph', {}).get('nodes', []))}\n")

    print("=== SUCCESS: All local server endpoints verified successfully! ===")

if __name__ == "__main__":
    test_endpoints()
