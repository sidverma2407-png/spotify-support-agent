import pytest
from src.generation.generator import ResponseGenerator
from src.generation.policy import EscalationPolicy
from src.generation.safety import SafetyChecker
from src.generation.templates import TemplateGenerator

@pytest.fixture
def generator():
    return ResponseGenerator()

def test_successful_generation(generator):
    evidence = [{"conversation_id": "1", "score": 0.85, "support_response": "Hi John! We fixed this. /AB", "intent": "playback_app_issue"}]
    res = generator.generate_response("App crashing", "playback_app_issue", evidence, 0.9)
    assert res['action'] == "auto_handle"
    assert "John" not in res['response']
    assert "/AB" not in res['response']

def test_missing_evidence(generator):
    res = generator.generate_response("App crashing", "playback_app_issue", [], 0.9)
    assert res['action'] == "escalate"
    assert "No historical examples" in res['reason']

def test_weak_retrieval(generator):
    evidence = [{"conversation_id": "1", "score": 0.50, "support_response": "Hi", "intent": "playback_app_issue"}]
    res = generator.generate_response("App crashing", "playback_app_issue", evidence, 0.9)
    assert res['action'] == "escalate"
    assert "below threshold" in res['reason']

def test_other_unclear_escalation(generator):
    evidence = [{"conversation_id": "1", "score": 0.90, "support_response": "Hi", "intent": "other_unclear"}]
    res = generator.generate_response("asdfasdf", "other_unclear", evidence, 0.9)
    assert res['action'] == "escalate"
    assert "unclear" in res['reason'].lower()

def test_sensitive_escalation(generator):
    evidence = [{"conversation_id": "1", "score": 0.90, "support_response": "Hi", "intent": "account_login"}]
    res = generator.generate_response("my account was hacked", "account_login", evidence, 0.9)
    assert res['action'] == "escalate"
    assert "hacked" in res['reason'].lower()

def test_unsupported_promises_escalation():
    # If the generation outputs a promise, it escalates
    # We can fake it by passing a raw uncleaned string that contains a promise
    evidence = [{"conversation_id": "1", "score": 0.90, "support_response": "we will refund you immediately", "intent": "playback_app_issue"}]
    gen = ResponseGenerator()
    res = gen.generate_response("query", "playback_app_issue", evidence, 0.9)
    assert res['action'] == "escalate"
    assert "Safety/Grounding violation" in res['reason']

def test_structured_output(generator):
    evidence = [{"conversation_id": "1", "score": 0.85, "support_response": "Hi", "intent": "playback_app_issue"}]
    res = generator.generate_response("query", "playback_app_issue", evidence, 0.9)
    assert "response" in res
    assert "action" in res
    assert "intent" in res
    assert "confidence" in res
    assert "evidence" in res
    assert "reason" in res
