import json
import csv
from pathlib import Path
from src.gold.sample_gold_set import sample_gold_set
from src.retrieval.evaluate_retrieval import get_train_test_split

def test_sampling_reproducibility():
    s1 = sample_gold_set(seed=42, target_size=18, out_file="data/gold_test/test_cands.json")
    s2 = sample_gold_set(seed=42, target_size=18, out_file="data/gold_test/test_cands.json")
    
    ids1 = [x['conversation_id'] for x in s1]
    ids2 = [x['conversation_id'] for x in s2]
    
    assert ids1 == ids2
    assert len(s1) == 18

def test_gold_train_separation():
    docs = []
    with open("data/processed/retrieval_corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line))
            
    train_corpus, _ = get_train_test_split(docs, random_state=42)
    train_ids = {x['conversation_id'] for x in train_corpus}
    
    sampled = sample_gold_set(seed=42, target_size=18, out_file="data/gold_test/test_cands.json")
    sampled_ids = {x['conversation_id'] for x in sampled}
    
    # Must be disjoint
    assert len(train_ids.intersection(sampled_ids)) == 0

def test_gold_csv_schema():
    csv_file = Path("data/gold_test/gold_eval.csv")
    if csv_file.exists():
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected = [
                "example_id", "conversation_id", "customer_message", 
                "gold_intent", "should_auto_handle", "response_quality_target", 
                "annotator_notes", "annotator", "timestamp"
            ]
            assert header == expected
