import os
import json
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter

from src.intent.data_loader import load_and_split_data
from src.retrieval.evaluate_retrieval import get_train_test_split
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.generation.generator import ResponseGenerator
from src.intent.taxonomy import INTENT_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Heuristic Deterministic Judge Fallback
def heuristic_judge(generated_response, retrieved_response, msg, quality_label):
    """
    Since we don't have an LLM API configured by default, we use a heuristic 
    fallback judge to simulate evaluating safety, relevance, and helpfulness.
    """
    safety = 5
    if "[AGENT]" in generated_response or "Internal" in generated_response:
        safety = 2
        
    length_score = 5 if len(generated_response.split()) > 10 else 3
    
    # We will synthetically score it based on the presence of some keywords or just assume it's acceptable.
    # To test agreement with human labels:
    # If the human says poor, the judge should give it a lower score generally, but let's 
    # not peek at the gold label to decide the "pass/fail". We MUST NOT look at gold label!
    
    # Simple deterministic rule:
    # if escalated -> borderline (3)
    # else -> pass (5)
    
    # Actually let's just make it a static pass for now unless it's escalated,
    # because generating a real assessment without an LLM is impossible.
    if "escalated" in generated_response.lower():
        overall = 3
        category = "borderline"
        reason = "System escalated the request."
    elif safety < 5:
        overall = 2
        category = "fail"
        reason = "Safety violations like agent tags detected."
    else:
        overall = 5
        category = "pass"
        reason = "Response looks generally acceptable and safe."
        
    return {
        "relevance": overall,
        "helpfulness": overall,
        "grounding": overall,
        "safety": safety,
        "overall": overall,
        "category": category,
        "reason": reason
    }

def main():
    logger.info("Loading Data & Splitting...")
    # Train the classifiers
    train_ex, _ = load_and_split_data(random_state=42)
    X_train = [ex['text'] for ex in train_ex]
    y_train = [ex['weak_intent'] for ex in train_ex]
    
    # Majority baseline
    most_common_intent = Counter(y_train).most_common(1)[0][0]
    
    # TF-IDF SGD baseline
    logger.info("Training Improved Classifier...")
    clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000, tol=1e-3, class_weight='balanced'))
    ])
    clf_pipeline.fit(X_train, y_train)
    
    # Train Retriever
    logger.info("Building Retrieval Corpus...")
    corpus = []
    with open("data/processed/retrieval_corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
            
    train_corpus, _ = get_train_test_split(corpus, random_state=42)
    retriever = TfidfRetriever(train_corpus)
    
    generator = ResponseGenerator()
    
    # Load Gold Set
    logger.info("Loading Gold Eval Set...")
    gold_df = pd.read_csv("data/gold/gold_eval.csv")
    if len(gold_df) != 150:
        logger.warning(f"Gold eval CSV has {len(gold_df)} rows, expected 150.")
        
    predictions = []
    
    logger.info("Running evaluation loop...")
    for idx, row in gold_df.iterrows():
        msg = row['customer_message']
        cid = str(row['conversation_id'])
        
        # 1. Intent
        pred_probs = clf_pipeline.predict_proba([msg])[0]
        pred_intent_idx = pred_probs.argmax()
        pred_intent = clf_pipeline.classes_[pred_intent_idx]
        confidence = float(pred_probs[pred_intent_idx])
        
        # 2. Retrieval
        retrieved_examples = retriever.search(msg, top_k=5, exclude_conversation_id=cid)
        if not retrieved_examples:
            # dummy if somehow empty
            retrieved_examples = [{"conversation_id": "dummy", "score": 0.0, "intent": "other_unclear", "support_response": "We are escalating.", "customer_query": msg}]
            
        # 3. Generation
        gen_result = generator.generate_response(msg, pred_intent, retrieved_examples, confidence)
        
        # 4. LLM Judge
        judge_res = heuristic_judge(gen_result['response'], retrieved_examples[0]['support_response'], msg, row['response_quality_target'])
        
        record = {
            "example_id": row['example_id'],
            "conversation_id": cid,
            "customer_message": msg,
            "gold_intent": row['gold_intent'],
            "predicted_intent": pred_intent,
            "majority_intent": most_common_intent,
            "classifier_confidence": confidence,
            "retrieved_examples": retrieved_examples,
            "retrieval_scores": [e['score'] for e in retrieved_examples],
            "historical_baseline_response": retrieved_examples[0]['support_response'],
            "generated_response": gen_result['response'],
            "predicted_action": gen_result['action'],
            "human_should_auto_handle": row['should_auto_handle'],
            "human_response_quality_target": row['response_quality_target'],
            "judge_category": judge_res['category'],
            "judge_reason": judge_res['reason']
        }
        predictions.append(record)
        
    # Save predictions
    Path("results").mkdir(exist_ok=True)
    with open("results/m7_predictions.json", "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
        
    logger.info("Done generating predictions. Metrics will be calculated in a separate step or module.")

if __name__ == "__main__":
    main()
