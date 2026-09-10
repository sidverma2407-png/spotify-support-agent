import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingRetriever:
    def __init__(self, corpus, model_name="all-MiniLM-L6-v2", cache_file="data/processed/embeddings_cache.npy"):
        """
        corpus: list of dicts with 'conversation_id', 'customer_query', 'support_response', 'intent'
        """
        self.corpus = corpus
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        cache_path = Path(cache_file)
        
        if cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            self.doc_vectors = np.load(cache_path)
            if len(self.doc_vectors) != len(corpus):
                logger.warning("Cache size mismatch, regenerating embeddings...")
                self._generate_and_cache(cache_path)
        else:
            self._generate_and_cache(cache_path)
            
    def _generate_and_cache(self, cache_path):
        logger.info(f"Generating embeddings using {self.model_name}...")
        # Index ONLY the customer query text to prevent response leakage
        texts = [doc['customer_query'] for doc in self.corpus]
        # Generate embeddings (batch_size optimized for CPU)
        self.doc_vectors = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        
        logger.info(f"Saving embeddings to {cache_path}")
        np.save(cache_path, self.doc_vectors)
        
    def search(self, query: str, top_k: int = 5, exclude_conversation_id: str = None):
        q_vec = self.model.encode([query])
        
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
