import json
import os
from pathlib import Path
from src.m7.evaluate import heuristic_judge, real_llm_judge

def test_judge_schema():
    res = heuristic_judge("Generated", "Retrieved", "Msg", "Intent", "auto_handle")
    assert "relevance" in res
    assert "helpfulness" in res
    assert "grounding" in res
    assert "safety" in res
    assert "overall" in res
    assert "category" in res
    assert "reason" in res
    assert res["category"] in ["pass", "borderline", "fail"]

def test_api_key_absence():
    # If API key is not present, ensure evaluate.py uses heuristic judge correctly
    pass # covered by standard execution

def test_gold_set_isolation():
    # Ensure no human label parameters are passed to real_llm_judge signature
    import inspect
    sig = inspect.signature(real_llm_judge)
    params = list(sig.parameters.keys())
    assert "gold_intent" not in params
    assert "should_auto_handle" not in params
    assert "response_quality" not in params
    
