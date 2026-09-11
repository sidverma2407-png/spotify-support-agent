import json
import csv
import sys
from datetime import datetime
from pathlib import Path

INTENTS = {
    "1": "account_login", "2": "billing_subscription", "3": "playback_app_issue", 
    "4": "content_availability", "5": "ads_recommendations", "6": "cancellation_refund", 
    "7": "agent_handoff", "8": "praise_gratitude", "9": "other_unclear"
}

def load_progress(csv_path, annotator):
    labeled_ids = set()
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("annotator") == annotator:
                    labeled_ids.add(row["conversation_id"])
    return labeled_ids

def run_cli(annotator="human_1"):
    candidates_file = Path("data/gold/gold_candidates.json")
    csv_path = Path("data/gold/gold_eval.csv")
    
    if not candidates_file.exists():
        print("Run sample_gold_set.py first.")
        return
        
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
        
    labeled_ids = load_progress(csv_path, annotator)
    
    total = len(candidates)
    remaining = [c for c in candidates if c['conversation_id'] not in labeled_ids]
    
    if not remaining:
        print(f"All {total} examples labeled! Great job.")
        return
        
    print(f"--- M6 Annotation CLI ---")
    print(f"Progress: {total - len(remaining)} / {total} labeled.\n")
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for doc in remaining:
            print("="*60)
            print(f"ID: {doc['conversation_id']}")
            print(f"MODEL PREDICTION (WEAK) — NOT GOLD LABEL: {doc['intent']}")
            print("-" * 60)
            print(f"MESSAGE: \n{doc['customer_query']}")
            print("="*60)
            
            # 1. Intent
            while True:
                print("\nIntents:")
                for k, v in INTENTS.items():
                    print(f"  {k}: {v}")
                val = input("Select Gold Intent (1-9) or 'q' to quit: ").strip()
                if val.lower() == 'q':
                    return
                if val in INTENTS:
                    gold_intent = INTENTS[val]
                    break
                print("Invalid selection.")
                
            # 2. Auto Handle
            while True:
                val = input("Should this be auto-handled? (y/n): ").strip().lower()
                if val in ['y', 'n']:
                    auto_handle = 'yes' if val == 'y' else 'no'
                    break
                print("Invalid selection (y/n).")
                
            # 3. Quality Target
            while True:
                val = input("Response Quality Target (g=good, a=acceptable, p=poor, n=not_applicable): ").strip().lower()
                mapping = {'g': 'good', 'a': 'acceptable', 'p': 'poor', 'n': 'not_applicable'}
                if val in mapping:
                    quality = mapping[val]
                    break
                print("Invalid selection.")
                
            # 4. Notes
            notes = input("Optional notes (press enter to skip): ").strip()
            
            writer.writerow([
                doc['conversation_id'], doc['conversation_id'], doc['customer_query'],
                gold_intent, auto_handle, quality, notes, annotator, datetime.now().isoformat()
            ])
            print(f"-> Saved {doc['conversation_id']}\n")

if __name__ == "__main__":
    annotator = sys.argv[1] if len(sys.argv) > 1 else "human_1"
    run_cli(annotator)
