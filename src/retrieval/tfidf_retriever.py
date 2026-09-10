import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, corpus):
        """
        corpus: list of dicts with 'conversation_id', 'customer_query', 'support_response', 'intent'
        """
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        
        # We index ONLY the customer query text to prevent response leakage
        texts = [doc['customer_query'] for doc in corpus]
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
    def search(self, query: str, top_k: int = 5, exclude_conversation_id: str = None):
        q_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity with all documents
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        
        # Sort indices by similarity descending
        ranked_indices = np.argsort(sims)[::-1]
        
        results = []
        for idx in ranked_indices:
            doc = self.corpus[idx]
            
            # Leakage prevention: Never retrieve from the same conversation
            if exclude_conversation_id and doc['conversation_id'] == exclude_conversation_id:
                continue
                
            results.append({
                "conversation_id": doc['conversation_id'],
                "score": float(sims[idx]),
                "customer_query": doc['customer_query'],
                "support_response": doc['support_response'],
                "intent": doc['intent']
            })
            
            if len(results) >= top_k:
                break
                
        return results
