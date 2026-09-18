"""
test_intent_filtering.py
Verification script for intent-specific answer filtering (answering only what was asked).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_agent import ask_agent, _detect_question_intent

def test_intent_detection():
    print("--- 1. Testing Intent Detection ---")
    assert _detect_question_intent("When should I take Ibuprocillin?") == "when_to_take"
    assert _detect_question_intent("Why should I use Ibuprocillin?") == "why_use"
    assert _detect_question_intent("What is its dosage form?") == "dosage"
    assert _detect_question_intent("Who is the manufacturer of Ibuprocillin?") == "manufacturer"
    assert _detect_question_intent("What precautions should I take?") == "precautions"
    print("[OK] Intent Detection passed.\n")

def test_intent_filtered_responses():
    print("--- 2. Testing Intent-Filtered Responses ---")
    
    # 1. When should I take Ibuprocillin?
    q_when = "When should I take Ibuprocillin?"
    resp_when = ask_agent(q_when)
    safe_when = resp_when.encode('ascii', 'ignore').decode('ascii')
    print("Response for 'When should I take Ibuprocillin?':\n" + safe_when + "\n")
    
    assert "When Should You Take It" in resp_when or "Prescribed" in resp_when
    assert "Doctor AI Clinical Assessment (Source:" not in resp_when
    assert "Hypertension" not in resp_when, "Irrelevant Hypertension guideline leaked"
    assert "Manufacturer:" not in resp_when, "Manufacturer leaked into 'when to take' question"

    # 2. Why should I use Ibuprocillin?
    q_why = "Why should I use Ibuprocillin?"
    resp_why = ask_agent(q_why)
    safe_why = resp_why.encode('ascii', 'ignore').decode('ascii')
    print("Response for 'Why should I use Ibuprocillin?':\n" + safe_why + "\n")

    assert "Why Use It" in resp_why or "Prescribed" in resp_why
    assert "Manufacturer:" not in resp_why

    print("[OK] Intent-Filtered Responses passed.\n")

if __name__ == "__main__":
    test_intent_detection()
    test_intent_filtered_responses()
    print("=== SUCCESS: All intent-specific answer filtering tests passed successfully! ===")
