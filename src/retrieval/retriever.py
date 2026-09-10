import json
from pathlib import Path
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.embedding_retriever import EmbeddingRetriever

class HistoricalRetriever:
    def __init__(self, method="embedding", corpus_path="data/processed/retrieval_corpus.jsonl"):
        self.method = method
        self.corpus = self._load_corpus(corpus_path)
        
        if method == "tfidf":
            self.backend = TfidfRetriever(self.corpus)
        elif method == "embedding":
            self.backend = EmbeddingRetriever(self.corpus)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
            
    def _load_corpus(self, path):
        docs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                docs.append(json.loads(line))
        return docs
        
    def search(self, query: str, top_k: int = 5, exclude_conversation_id: str = None):
        """
        Search for relevant historical conversations.
        
        Args:
            query: The customer's message to search for.
            top_k: Number of results to return.
            exclude_conversation_id: If provided, prevents retrieving any messages 
                                     from this specific conversation (Leakage Prevention).
                                     
        Returns:
            List of dictionaries containing conversation_id, score, customer_query, support_response, intent
        """
        return self.backend.search(query, top_k=top_k, exclude_conversation_id=exclude_conversation_id)
