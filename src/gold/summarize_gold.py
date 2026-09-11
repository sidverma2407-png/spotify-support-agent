import csv
import json
from pathlib import Path
from collections import Counter

def summarize():
    csv_path = Path("data/gold/gold_eval.csv")
    candidates_path = Path("data/gold/gold_candidates.json")
    
    if not candidates_path.exists():
        print("Gold set not initialized.")
        return
        
    with open(candidates_path, 'r', encoding='utf-8') as f:
        target_size = len(json.load(f))
        
    labels = []
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            labels = list(reader)
            
    completed = len(labels)
    intents = Counter([row['gold_intent'] for row in labels])
    auto = Counter([row['should_auto_handle'] for row in labels])
    qual = Counter([row['response_quality_target'] for row in labels])
    
    # Calculate double-labeled (agreement)
    by_id = {}
    double_labeled = 0
    agreements = 0
    
    for row in labels:
        cid = row['conversation_id']
        if cid not in by_id:
            by_id[cid] = row
        else:
            double_labeled += 1
            if by_id[cid]['gold_intent'] == row['gold_intent']:
                agreements += 1
                
    summary = {
        "target_size": target_size,
        "completed": completed,
        "is_fully_labeled": completed >= target_size,
        "intents": dict(intents),
        "auto_handle": dict(auto),
        "quality": dict(qual),
        "double_labeled_count": double_labeled,
        "raw_agreement_rate": (agreements / double_labeled) if double_labeled > 0 else None
    }
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "gold_set_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    with open(out_dir / "gold_set_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Gold Evaluation Set Summary\n\n")
        status = "Complete" if summary['is_fully_labeled'] else "In Progress (Pending Human Annotation)"
        f.write(f"**Status:** {status}\n\n")
        f.write(f"- **Target Size:** {target_size}\n")
        f.write(f"- **Completed Annotations:** {completed}\n")
        f.write(f"- **Double-Labeled:** {double_labeled}\n")
        if double_labeled > 0:
            f.write(f"- **Raw Agreement Rate:** {summary['raw_agreement_rate']*100:.1f}%\n")
            
        f.write("\n## Current Label Distributions\n")
        if completed == 0:
            f.write("*No examples have been manually labeled yet.*\n")
        else:
            f.write("\n### Gold Intents\n")
            for k, v in intents.items():
                f.write(f"- {k}: {v}\n")
                
if __name__ == "__main__":
    summarize()
