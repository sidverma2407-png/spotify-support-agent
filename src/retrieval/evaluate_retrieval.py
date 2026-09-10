import json
import random
import time
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_train_test_split(corpus, test_size=0.2, random_state=42):
    conv_ids = sorted(list({doc['conversation_id'] for doc in corpus}))
    random.seed(random_state)
    random.shuffle(conv_ids)
    
    split_idx = int(len(conv_ids) * (1 - test_size))
    train_convs = set(conv_ids[:split_idx])
    test_convs = set(conv_ids[split_idx:])
    
    assert len(train_convs.intersection(test_convs)) == 0, "Leakage detected!"
    
    train_corpus = [doc for doc in corpus if doc['conversation_id'] in train_convs]
    test_corpus = [doc for doc in corpus if doc['conversation_id'] in test_convs]
    
    return train_corpus, test_corpus

def evaluate_method(method_name, train_corpus, test_corpus):
    logger.info(f"--- Evaluating {method_name} ---")
    start_time = time.time()
    
    if method_name == 'embedding':
        cache_eval = "data/processed/embeddings_eval_cache.npy"
        from src.retrieval.embedding_retriever import EmbeddingRetriever
        backend = EmbeddingRetriever(train_corpus, cache_file=cache_eval)
    else:
        from src.retrieval.tfidf_retriever import TfidfRetriever
        backend = TfidfRetriever(train_corpus)
        
    idx_time = time.time() - start_time
    logger.info(f"Indexing took {idx_time:.2f} seconds")
    
    random.seed(42)
    eval_subset = random.sample(test_corpus, min(1000, len(test_corpus)))
    
    metrics = {
        'exact_recall@1': 0, 'exact_recall@3': 0, 'exact_recall@5': 0, 'exact_recall@10': 0, 'exact_mrr': 0.0,
        'intent_recall@1': 0, 'intent_recall@3': 0, 'intent_recall@5': 0, 'intent_recall@10': 0, 'intent_mrr': 0.0
    }
    
    query_times = []
    qualitative_examples = []
    
    for i, doc in enumerate(eval_subset):
        q_start = time.time()
        res = backend.search(doc['customer_query'], top_k=10, exclude_conversation_id=doc['conversation_id'])
        query_times.append(time.time() - q_start)
        
        target_conv_id = doc['conversation_id']
        target_intent = doc['intent']
        
        exact_ranks = [j+1 for j, r in enumerate(res) if r['conversation_id'] == target_conv_id]
        if exact_ranks:
            rank = exact_ranks[0]
            if rank <= 1: metrics['exact_recall@1'] += 1
            if rank <= 3: metrics['exact_recall@3'] += 1
            if rank <= 5: metrics['exact_recall@5'] += 1
            if rank <= 10: metrics['exact_recall@10'] += 1
            metrics['exact_mrr'] += 1.0 / rank
            
        intent_ranks = [j+1 for j, r in enumerate(res) if r['intent'] == target_intent]
        if intent_ranks:
            rank = intent_ranks[0]
            if rank <= 1: metrics['intent_recall@1'] += 1
            if rank <= 3: metrics['intent_recall@3'] += 1
            if rank <= 5: metrics['intent_recall@5'] += 1
            if rank <= 10: metrics['intent_recall@10'] += 1
            metrics['intent_mrr'] += 1.0 / rank
            
        if method_name == "embedding" and len(qualitative_examples) < 10 and res:
            if doc['intent'] not in [e['query_intent'] for e in qualitative_examples]:
                qualitative_examples.append({
                    "incoming_query": doc['customer_query'],
                    "query_intent": doc['intent'],
                    "top_retrieved_query": res[0]['customer_query'],
                    "top_retrieved_response": res[0]['support_response'],
                    "retrieved_intent": res[0]['intent'],
                    "score": res[0]['score'],
                    "intent_match_proxy": "Match" if res[0]['intent'] == doc['intent'] else "Mismatch",
                    "short_explanation": f"Retrieved {res[0]['intent']} vs requested {doc['intent']}"
                })
            elif i > 500 and len(qualitative_examples) < 10:
                qualitative_examples.append({
                    "incoming_query": doc['customer_query'],
                    "query_intent": doc['intent'],
                    "top_retrieved_query": res[0]['customer_query'],
                    "top_retrieved_response": res[0]['support_response'],
                    "retrieved_intent": res[0]['intent'],
                    "score": res[0]['score'],
                    "intent_match_proxy": "Match" if res[0]['intent'] == doc['intent'] else "Mismatch",
                    "short_explanation": f"Retrieved {res[0]['intent']} vs requested {doc['intent']}"
                })
            
    n = len(eval_subset)
    for k in metrics:
        metrics[k] /= n
        
    metrics['index_time_sec'] = idx_time
    metrics['avg_query_time_sec'] = sum(query_times) / n
    
    if qualitative_examples:
        with open("results/retrieval_examples.json", "w", encoding='utf-8') as f:
            json.dump(qualitative_examples, f, indent=2)
    
    return metrics

def main():
    docs = []
    with open("data/processed/retrieval_corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line))
            
    train_corpus, test_corpus = get_train_test_split(docs)
    
    results = {}
    
    res_tfidf = evaluate_method("tfidf", train_corpus, test_corpus)
    results["tfidf"] = res_tfidf
    
    res_emb = evaluate_method("embedding", train_corpus, test_corpus)
    results["embedding"] = res_emb
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "retrieval_metrics.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "retrieval_metrics.md", "w", encoding='utf-8') as f:
        f.write("# Retrieval Evaluation\n\n")
        f.write("## Weak-Label Intent Agreement\n")
        f.write("*(Note: This metric measures whether the retrieved historical conversation shares the same weak intent label as the test query. It is a diagnostic proxy, NOT a true human relevance judgment.)*\n")
        f.write("| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(f"| TF-IDF | {res_tfidf['intent_recall@1']:.3f} | {res_tfidf['intent_recall@3']:.3f} | {res_tfidf['intent_recall@5']:.3f} | {res_tfidf['intent_recall@10']:.3f} | {res_tfidf['intent_mrr']:.3f} |\n")
        f.write(f"| Embedding | {res_emb['intent_recall@1']:.3f} | {res_emb['intent_recall@3']:.3f} | {res_emb['intent_recall@5']:.3f} | {res_emb['intent_recall@10']:.3f} | {res_emb['intent_mrr']:.3f} |\n\n")
        
        f.write("## Exact Historical Conversation Retrieval\n")
        f.write("*(Note: Expected to be 0.0 because of strict conversation-level train/test isolation, preventing self-retrieval)*\n")
        f.write("| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(f"| TF-IDF | {res_tfidf['exact_recall@1']:.3f} | {res_tfidf['exact_recall@3']:.3f} | {res_tfidf['exact_recall@5']:.3f} | {res_tfidf['exact_recall@10']:.3f} | {res_tfidf['exact_mrr']:.3f} |\n")
        f.write(f"| Embedding | {res_emb['exact_recall@1']:.3f} | {res_emb['exact_recall@3']:.3f} | {res_emb['exact_recall@5']:.3f} | {res_emb['exact_recall@10']:.3f} | {res_emb['exact_mrr']:.3f} |\n\n")

if __name__ == "__main__":
    main()