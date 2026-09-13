
import pytest
from src.m7.evaluate import heuristic_judge, real_llm_judge

def test_heuristic_judge_safe_auto_handle():
    res = heuristic_judge('Here is a normal response', 'Past response', 'msg', 'intent', 'auto_handle')
    assert res['safety'] == 5
    assert res['category'] == 'pass'
    assert res['judge_type'] == 'heuristic'

def test_heuristic_judge_escalation():
    res = heuristic_judge('This request has been escalated to a human agent.', 'Past response', 'msg', 'intent', 'escalate')
    assert res['category'] == 'borderline'
    assert res['overall'] == 3
    assert res['judge_type'] == 'heuristic'

def test_heuristic_judge_safety_violation():
    res = heuristic_judge('Here is a bad response with [AGENT] tag', 'Past response', 'msg', 'intent', 'auto_handle')
    assert res['safety'] == 2
    assert res['category'] == 'fail'
    assert res['judge_type'] == 'heuristic'

def test_real_llm_judge_missing_key():
    # Will fallback to heuristic because no key is provided
    res = real_llm_judge('Generated', 'Retrieved', 'Msg', 'Intent', 'auto_handle', 'invalid_key')
    assert res['judge_type'] == 'heuristic'
    assert 'Heuristic Fallback' in res['reason']

