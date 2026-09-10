import pytest
from src.retrieval.tfidf_retriever import TfidfRetriever

@pytest.fixture
def dummy_corpus():
    return [
        {"conversation_id": "c1", "customer_query": "hello world", "support_response": "hi", "intent": "praise_gratitude"},
        {"conversation_id": "c2", "customer_query": "how to refund", "support_response": "here", "intent": "cancellation_refund"},
        {"conversation_id": "c3", "customer_query": "app crashing", "support_response": "update", "intent": "playback_app_issue"},
        {"conversation_id": "c4", "customer_query": "hello again", "support_response": "hi again", "intent": "praise_gratitude"},
    ]

def test_top_k_retrieval(dummy_corpus):
    retriever = TfidfRetriever(dummy_corpus)
    res = retriever.search("hello", top_k=2)
    assert len(res) == 2
    assert res[0]['conversation_id'] in ["c1", "c4"]
    assert res[1]['conversation_id'] in ["c1", "c4"]

def test_empty_short_query(dummy_corpus):
    retriever = TfidfRetriever(dummy_corpus)
    res = retriever.search("", top_k=2)
    assert len(res) <= 2  # TF-IDF might return 0 matches for empty string
    
    res2 = retriever.search("a", top_k=2)
    assert len(res2) <= 2

def test_duplicate_results(dummy_corpus):
    retriever = TfidfRetriever(dummy_corpus)
    res = retriever.search("hello", top_k=5)
    c_ids = [r['conversation_id'] for r in res]
    assert len(c_ids) == len(set(c_ids)), "Duplicate results returned"

def test_conversation_level_isolation(dummy_corpus):
    retriever = TfidfRetriever(dummy_corpus)
    
    # Query is exactly c1's query, but we exclude c1
    res = retriever.search("hello world", top_k=5, exclude_conversation_id="c1")
    c_ids = [r['conversation_id'] for r in res]
    
    assert "c1" not in c_ids, "Leakage detected! Retrieved the excluded conversation."

def test_correct_result_schema(dummy_corpus):
    retriever = TfidfRetriever(dummy_corpus)
    res = retriever.search("hello", top_k=1)
    
    assert len(res) > 0
    doc = res[0]
    
    expected_keys = {"conversation_id", "score", "customer_query", "support_response", "intent"}
    assert set(doc.keys()) == expected_keys
    assert isinstance(doc['score'], float)
