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
import time

from src.intent.data_loader import load_and_split_data
from src.retrieval.evaluate_retrieval import get_train_test_split
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.generation.generator import ResponseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = Path("results/judge_cache.json")

def load_judge_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_judge_cache(cache):
    Path("results").mkdir(exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def real_llm_judge(generated_response, retrieved_response, msg, pred_intent, pred_action, api_key, model="gpt-4o-mini"):
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""You are an objective judge evaluating a customer support AI.
DO NOT use chain-of-thought. Output ONLY a valid JSON object.

Customer Message: {msg}
Predicted Intent: {pred_intent}
Predicted Action: {pred_action}
Retrieved Historical Evidence: {retrieved_response}
AI Generated Response: {generated_response}

Evaluate the generated response. Score 1-5 for Relevance, Helpfulness, Grounding, Safety, Overall.
Categorize as 'pass', 'borderline', or 'fail'. Provide a short reason.
JSON schema: {{"relevance": int, "helpfulness": int, "grounding": int, "safety": int, "overall": int, "category": "pass"|"borderline"|"fail", "reason": "short text"}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM Judge error: {e}")
        return heuristic_judge(generated_response, retrieved_response, msg, pred_intent, pred_action)

def heuristic_judge(generated_response, retrieved_response, msg, pred_intent, pred_action):
    safety = 5
    if "[AGENT]" in generated_response or "Internal" in generated_response:
        safety = 2
        
    if "escalated" in generated_response.lower() or pred_action == 'escalate':
        overall = 3
        category = "borderline"
        reason = "System escalated the request."
    elif safety < 5:
        overall = 2
        category = "fail"
        reason = "Safety violations detected."
    else:
        overall = 5
        category = "pass"
        reason = "Response looks generally acceptable."
        
    return {
        "relevance": overall,
        "helpfulness": overall,
        "grounding": overall,
        "safety": safety,
        "overall": overall,
        "category": category,
        "reason": f"Heuristic Judge: {reason}"
    }

def main():
    logger.info("Starting M7 Evaluation...")
    
    gold_df = pd.read_csv("data/gold/gold_eval.csv", dtype=str)
    if len(gold_df) == 0:
        logger.error("gold_eval.csv is empty! Human annotation required via Streamlit UI.")
        return
        
    # Check API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    judge_model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    if api_key:
        logger.info(f"Using REAL LLM Judge ({judge_model})")
        judge_info = {"provider": "openai", "model": judge_model, "is_heuristic": False}
    else:
        logger.warning("No OPENAI_API_KEY found. Using Deterministic Heuristic Fallback Judge.")
        judge_info = {"provider": "local", "model": "heuristic", "is_heuristic": True}

    judge_cache = load_judge_cache()
    
    # Train classifiers & retriever
    train_ex, _ = load_and_split_data(random_state=42)
    X_train = [ex['text'] for ex in train_ex]
    y_train = [ex['weak_intent'] for ex in train_ex]
    most_common_intent = Counter(y_train).most_common(1)[0][0]
    
    clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000, tol=1e-3, class_weight='balanced'))
    ])
    clf_pipeline.fit(X_train, y_train)
    
    corpus = []
    with open("data/processed/retrieval_corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
            
    train_corpus, _ = get_train_test_split(corpus, random_state=42)
    retriever = TfidfRetriever(train_corpus)
    generator = ResponseGenerator()
    
    predictions = []
    for idx, row in gold_df.iterrows():
        msg = row['customer_message']
        cid = str(row['conversation_id'])
        
        pred_probs = clf_pipeline.predict_proba([msg])[0]
        pred_intent_idx = pred_probs.argmax()
        pred_intent = clf_pipeline.classes_[pred_intent_idx]
        confidence = float(pred_probs[pred_intent_idx])
        
        retrieved_examples = retriever.search(msg, top_k=5, exclude_conversation_id=cid)
        if not retrieved_examples:
            retrieved_examples = [{"conversation_id": "dummy", "score": 0.0, "intent": "other_unclear", "support_response": "We are escalating.", "customer_query": msg}]
            
        gen_result = generator.generate_response(msg, pred_intent, retrieved_examples, confidence)
        
        cache_key = f"{cid}_{judge_info['model']}"
        if cache_key in judge_cache:
            judge_res = judge_cache[cache_key]
        else:
            if not judge_info['is_heuristic']:
                judge_res = real_llm_judge(gen_result['response'], retrieved_examples[0]['support_response'], msg, pred_intent, gen_result['action'], api_key, judge_model)
            else:
                judge_res = heuristic_judge(gen_result['response'], retrieved_examples[0]['support_response'], msg, pred_intent, gen_result['action'])
            judge_cache[cache_key] = judge_res
            
        record = {
            "example_id": row.get('example_id', cid),
            "conversation_id": cid,
            "customer_message": msg,
            "gold_intent": row['gold_intent'],
            "predicted_intent": pred_intent,
            "majority_intent": most_common_intent,
            "classifier_confidence": confidence,
            "historical_baseline_response": retrieved_examples[0]['support_response'],
            "generated_response": gen_result['response'],
            "predicted_action": gen_result['action'],
            "human_should_auto_handle": row['should_auto_handle'],
            "human_response_quality_target": row['response_quality_target'],
            "judge_result": judge_res
        }
        predictions.append(record)
        
    save_judge_cache(judge_cache)
    
    # Calculate Metrics
    from sklearn.metrics import cohen_kappa_score
    gold_intents = [p['gold_intent'] for p in predictions]
    pred_intents = [p['predicted_intent'] for p in predictions]
    maj_intents = [p['majority_intent'] for p in predictions]
    
    acc_pred = accuracy_score(gold_intents, pred_intents)
    acc_maj = accuracy_score(gold_intents, maj_intents)
    _, _, f1, _ = precision_recall_fscore_support(gold_intents, pred_intents, average='macro', zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(gold_intents, pred_intents, average='weighted', zero_division=0)
    
    gold_action = ["auto_handle" if p['human_should_auto_handle'] == 'yes' else 'escalate' for p in predictions]
    pred_action = [p['predicted_action'] for p in predictions]
    action_acc = accuracy_score(gold_action, pred_action)
    
    # Human vs LLM mapping
    human_binary = ["pass" if p['human_response_quality_target'] in ['good', 'acceptable'] else "fail" for p in predictions]
    judge_binary = ["pass" if p['judge_result']['category'] in ['pass', 'borderline'] else "fail" for p in predictions]
    
    raw_agreement = accuracy_score(human_binary, judge_binary)
    
    try:
        kappa = cohen_kappa_score(human_binary, judge_binary)
    except:
        kappa = 0.0
        
    cm = confusion_matrix(human_binary, judge_binary, labels=["pass", "fail"]).tolist()
    
    metrics = {
        "judge_info": judge_info,
        "intent_accuracy": acc_pred,
        "intent_macro_f1": f1,
        "action_accuracy": action_acc,
        "raw_agreement": raw_agreement,
        "kappa": kappa,
        "confusion_matrix": {"labels": ["pass", "fail"], "matrix": cm}
    }
    
    with open("results/m7_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()
