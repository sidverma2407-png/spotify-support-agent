import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate_predictions(y_true, y_pred, labels, model_name, texts=None):
    acc = accuracy_score(y_true, y_pred)
    ma_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    ma_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    ma_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    wt_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": ma_p,
        "macro_recall": ma_r,
        "macro_f1": ma_f1,
        "weighted_f1": wt_f1,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": report[label]["support"]
            } for label in labels if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    # Save metrics JSON
    json_path = out_dir / f"intent_metrics_{model_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    # Save metrics Markdown
    md_path = out_dir / f"intent_metrics_{model_name}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Intent Classification Evaluation: {model_name}\n\n")
        f.write(f"- **Accuracy**: {acc:.4f}\n")
        f.write(f"- **Macro F1**: {ma_f1:.4f}\n")
        f.write(f"- **Weighted F1**: {wt_f1:.4f}\n\n")
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|---------|\n")
        for label in labels:
            if label in metrics["per_class"]:
                c_m = metrics["per_class"][label]
                f.write(f"| {label} | {c_m['precision']:.4f} | {c_m['recall']:.4f} | {c_m['f1-score']:.4f} | {c_m['support']} |\n")
                
    # Save representative errors if texts provided
    if texts is not None:
        errors = []
        for text, yt, yp in zip(texts, y_true, y_pred):
            if yt != yp:
                errors.append({"text": text, "true_label": yt, "predicted_label": yp})
        
        # Take a sample of up to 50 errors
        import random
        random.seed(42)
        sample_errors = random.sample(errors, min(50, len(errors)))
        
        error_path = out_dir / f"intent_errors_{model_name}.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(sample_errors, f, indent=2)
            
    return metrics
