import pytest
import json
import numpy as np
from src.intent.taxonomy import TAXONOMY, INTENT_CLASSES
from src.intent.data_loader import load_and_split_data
from src.intent.build_labels import weak_label

def test_taxonomy_validity():
    assert len(TAXONOMY) == 9
    assert "account_login" in INTENT_CLASSES
    assert "other_unclear" in INTENT_CLASSES

def test_deterministic_splitting():
    train1, test1 = load_and_split_data(random_state=42)
    train2, test2 = load_and_split_data(random_state=42)
    
    assert len(train1) == len(train2)
    assert len(test1) == len(test2)
    # Check that they are identically split
    assert train1[0]['conversation_id'] == train2[0]['conversation_id']

def test_no_conversation_leakage():
    train_ex, test_ex = load_and_split_data(random_state=42)
    
    train_convs = {ex['conversation_id'] for ex in train_ex}
    test_convs = {ex['conversation_id'] for ex in test_ex}
    
    intersection = train_convs.intersection(test_convs)
    assert len(intersection) == 0, f"Leakage found! {len(intersection)} overlapping conversations."

def test_weak_label_heuristics():
    assert weak_label("my account was hacked") == "account_login"
    assert weak_label("how to cancel my premium") == "cancellation_refund"
    assert weak_label("the app keeps crashing on windows") == "playback_app_issue"
    assert weak_label("is taylor swift available") == "content_availability"
    assert weak_label("thanks for the help!") == "praise_gratitude"
    assert weak_label("sent DM") == "agent_handoff"
    
    # Test context fallback
    assert weak_label("sure thing", prev_customer_text="the app keeps crashing") == "playback_app_issue"

def test_evaluation_metrics(tmp_path):
    from evaluation.intent_evaluation import evaluate_predictions
    y_true = ["account_login", "playback_app_issue", "account_login"]
    y_pred = ["account_login", "other_unclear", "account_login"]
    
    # Temporarily monkeypatch Path to use tmp_path for test isolation
    import evaluation.intent_evaluation
    original_path = evaluation.intent_evaluation.Path
    class MockPath(type(tmp_path)):
        def __new__(cls, *args, **kwargs):
            return tmp_path / args[0] if args else tmp_path
    
    evaluation.intent_evaluation.Path = MockPath
    
    try:
        metrics = evaluate_predictions(y_true, y_pred, INTENT_CLASSES, "test_model")
        assert metrics['accuracy'] == 2/3
        assert metrics['model'] == "test_model"
    finally:
        evaluation.intent_evaluation.Path = original_path
