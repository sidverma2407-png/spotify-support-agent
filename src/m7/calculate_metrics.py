import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    preds_file = Path("results/m7_predictions.json")
    if not preds_file.exists():
        logger.error("No predictions found!")
        return
        
    with open(preds_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
        
    # Intents
    gold_intents = [p['gold_intent'] for p in preds]
    pred_intents = [p['predicted_intent'] for p in preds]
    maj_intents = [p['majority_intent'] for p in preds]
    
    # Intent Metrics
    acc_pred = accuracy_score(gold_intents, pred_intents)
    acc_maj = accuracy_score(gold_intents, maj_intents)
    
    p, r, f1, _ = precision_recall_fscore_support(gold_intents, pred_intents, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(gold_intents, pred_intents, average='weighted', zero_division=0)
    
    # Action Metrics (auto_handle = positive class typically, but let's just do accuracy)
    # The gold is 'yes' or 'no' for auto-handle
    gold_action = ["auto_handle" if p['human_should_auto_handle'] == 'yes' else 'escalate' for p in preds]
    pred_action = [p['predicted_action'] for p in preds]
    
    action_acc = accuracy_score(gold_action, pred_action)
    
    # Precision/Recall for escalation (escalate = positive)
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        gold_action, pred_action, labels=['escalate'], average=None, zero_division=0
    )
    # Precision/Recall for auto-handle
    auto_p, auto_r, auto_f1, _ = precision_recall_fscore_support(
        gold_action, pred_action, labels=['auto_handle'], average=None, zero_division=0
    )
    
    escalation_rate = sum(1 for a in pred_action if a == 'escalate') / len(pred_action)
    
    # Judge vs Human
    # Human pass: good, acceptable
    # Judge pass: pass, borderline
    human_binary = ["pass" if p['human_response_quality_target'] in ['good', 'acceptable'] else "fail" for p in preds]
    judge_binary = ["pass" if p['judge_category'] in ['pass', 'borderline'] else "fail" for p in preds]
    
    agreement_acc = accuracy_score(human_binary, judge_binary)
    
    metrics = {
        "intent": {
            "accuracy": acc_pred,
            "macro_f1": f1,
            "weighted_f1": f1_w,
            "majority_baseline_accuracy": acc_maj
        },
        "action": {
            "accuracy": action_acc,
            "escalation_rate": escalation_rate,
            "auto_handle_precision": float(auto_p[0]) if len(auto_p) else 0.0,
            "auto_handle_recall": float(auto_r[0]) if len(auto_r) else 0.0,
            "escalate_precision": float(esc_p[0]) if len(esc_p) else 0.0,
            "escalate_recall": float(esc_r[0]) if len(esc_r) else 0.0,
        },
        "judge_agreement": {
            "accuracy": agreement_acc
        },
        "headline_metric": {
            "name": "Percentage of Examples Correctly Handled (Intent + Appropriate Action)",
            "value": sum(1 for p in preds if p['gold_intent'] == p['predicted_intent'] and 
                         ("auto_handle" if p['human_should_auto_handle'] == 'yes' else 'escalate') == p['predicted_action']) / len(preds)
        }
    }
    
    with open("results/m7_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    with open("results/m7_judge_results.json", "w", encoding='utf-8') as f:
        json.dump([{"example_id": p['example_id'], "judge_category": p['judge_category'], "reason": p['judge_reason']} for p in preds], f, indent=2)
        
    with open("results/m7_agreement.json", "w", encoding='utf-8') as f:
        json.dump({"raw_agreement": agreement_acc, "human_distribution": Counter(human_binary), "judge_distribution": Counter(judge_binary)}, f, indent=2)
        
    # Get 5 strong, 5 borderline, 5 failure
    strong = [p for p in preds if p['judge_category'] == 'pass'][:5]
    border = [p for p in preds if p['judge_category'] == 'borderline'][:5]
    fail = [p for p in preds if p['judge_category'] == 'fail'][:5]
    
    examples = {
        "strong": strong,
        "borderline": border,
        "failures": fail
    }
    
    with open("results/m7_examples.json", "w", encoding='utf-8') as f:
        json.dump(examples, f, indent=2)
        
    # Write MD
    md_content = f"""# M7 Evaluation Metrics

## Headline Metric
**{metrics['headline_metric']['name']}**: {metrics['headline_metric']['value']*100:.1f}%

*What is misleading about this metric?*
- The gold set is heavily skewed (due to manual user testing artifacts).
- It relies on only 150 examples from a single historical source (Twitter).
- Heuristic fallback judge is used instead of a real LLM.

## Intent Classification
- Accuracy: {acc_pred:.3f} (vs Majority: {acc_maj:.3f})
- Macro F1: {f1:.3f}
- Weighted F1: {f1_w:.3f}

## Escalation Policy
- Action Accuracy: {action_acc:.3f}
- System Escalation Rate: {escalation_rate:.3f}
- Auto-Handle Precision: {float(auto_p[0]) if len(auto_p) else 0.0:.3f} | Recall: {float(auto_r[0]) if len(auto_r) else 0.0:.3f}
- Escalate Precision: {float(esc_p[0]) if len(esc_p) else 0.0:.3f} | Recall: {float(esc_r[0]) if len(esc_r) else 0.0:.3f}

## Judge vs Human Agreement
- Binary Agreement Accuracy (Pass vs Fail): {agreement_acc:.3f}
- Note: Evaluated using heuristic deterministic fallback judge because no LLM API is provided.
"""
    with open("results/m7_metrics.md", "w", encoding='utf-8') as f:
        f.write(md_content)
        
    logger.info("Done calculating metrics.")

if __name__ == "__main__":
    main()
