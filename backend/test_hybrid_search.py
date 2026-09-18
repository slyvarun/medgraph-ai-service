"""
test_hybrid_search.py
Verification script for Hybrid Search (Graph + Dense Vector RRF).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import (
    ask_agent,
    _search_vector_guidelines,
    _reciprocal_rank_fusion,
    _get_clinical_guidelines
)

def test_vector_guideline_search():
    print("--- 1. Testing Vector Guideline Search ---")
    query = "What are the best long-term management strategies for diabetic neuropathy alongside medication?"
    results = _search_vector_guidelines(query)
    print(f"Retrieved {len(results)} guideline passage(s) for query: '{query}'")
    assert len(results) > 0, "Vector search failed to retrieve guideline passages"
    first = results[0]
    print(f"Top Guideline: {first.get('title')} ({first.get('condition')})")
    assert "Diabetic Neuropathy" in first.get("condition", ""), "Expected Diabetic Neuropathy guideline"
    print("[OK] Vector Guideline Search passed.\n")

def test_reciprocal_rank_fusion():
    print("--- 2. Testing Reciprocal Rank Fusion (RRF) ---")
    graph_records = [
        {"medicine": {"name": "Duloxetine", "category": "SNRI", "indication": "Diabetic Neuropathy"}},
        {"medicine": {"name": "Pregabalin", "category": "Gabapentinoid", "indication": "Diabetic Neuropathy"}}
    ]
    vector_guidelines = _get_clinical_guidelines()[:2]

    fused = _reciprocal_rank_fusion(graph_records, vector_guidelines, k=60)
    print(f"RRF Fused Count: {fused['total_fused']} (Graph: {len(fused['graph_records'])}, Vector: {len(fused['vector_guidelines'])})")
    assert len(fused["graph_records"]) == 2, "Expected 2 graph records in fused output"
    assert len(fused["vector_guidelines"]) == 2, "Expected 2 vector guidelines in fused output"
    print("[OK] Reciprocal Rank Fusion passed.\n")

def test_end_to_end_hybrid_query():
    print("--- 3. Testing End-to-End Hybrid RAG Query ---")
    query = "What are the best long-term management strategies for diabetic neuropathy alongside medication?"
    response = ask_agent(query)
    safe_response = response.encode('ascii', 'ignore').decode('ascii')
    print(f"RAG Response Preview:\n{safe_response[:400]}...\n")
    assert "Diabetic Neuropathy" in response or "Long-Term Management" in response, "Response missing clinical guideline context"
    assert "Glycemic Control" in response or "Foot" in response or "Clinical Management" in response, "Response missing non-pharmacological care strategies"
    print("[OK] End-to-End Hybrid RAG Query passed.\n")

if __name__ == "__main__":
    test_vector_guideline_search()
    test_reciprocal_rank_fusion()
    test_end_to_end_hybrid_query()
    print("=== SUCCESS: All Hybrid Search (Graph + Vector RRF) tests passed successfully! ===")
