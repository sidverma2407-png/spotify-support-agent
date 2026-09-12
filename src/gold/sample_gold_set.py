import json
import random
import os
import csv
from collections import defaultdict
from pathlib import Path
from src.retrieval.evaluate_retrieval import get_train_test_split

INTENTS = [
    "account_login", "billing_subscription", "playback_app_issue", 
    "content_availability", "ads_recommendations", "cancellation_refund", 
    "agent_handoff", "praise_gratitude", "other_unclear"
]

def sample_gold_set(seed=42, target_size=150, out_file="data/gold/gold_candidates.json"):
    random.seed(seed)
    
    docs = []
    with open("data/processed/retrieval_corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line))
            
    _, test_corpus = get_train_test_split(docs, random_state=seed)
    
    by_intent = defaultdict(list)
    for doc in test_corpus:
        by_intent[doc['intent']].append(doc)
        
    candidates = []
    
    # 150 / 9 = 16 with remainder 6.
    base_count = target_size // len(INTENTS)
    remainder = target_size % len(INTENTS)
    
    for i, intent in enumerate(INTENTS):
        pool = by_intent[intent]
        pool.sort(key=lambda x: x['conversation_id'])
        
        take = base_count + (1 if i < remainder else 0)
        sampled = random.sample(pool, min(take, len(pool)))
        candidates.extend(sampled)
        
    candidates.sort(key=lambda x: x['conversation_id'])
    
    out_file = Path(out_file)
    out_file.parent.mkdir(exist_ok=True)
    
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(candidates, f, indent=2)
        
    print(f"Sampled {len(candidates)} gold candidates (saved to {out_file}).")
    return candidates

def init_gold_csv():
    csv_file = Path("data/gold/gold_eval.csv")
    if not csv_file.exists():
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "example_id", "conversation_id", "customer_message", 
                "gold_intent", "should_auto_handle", "response_quality_target", 
                "annotator_notes", "annotator", "timestamp"
            ])

if __name__ == "__main__":
    sample_gold_set()
    init_gold_csv()
