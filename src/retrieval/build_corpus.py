import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_retrieval_corpus():
    labels_file = Path("data/processed/labeled_intent_data.jsonl")
    convs_file = Path("data/processed/spotify_conversations.jsonl")
    out_file = Path("data/processed/retrieval_corpus.jsonl")
    
    if not labels_file.exists() or not convs_file.exists():
        logger.error("Missing required data files.")
        return
        
    # Load intents
    intent_map = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            intent_map[data['tweet_id']] = data['weak_intent']
            
    docs = []
    
    with open(convs_file, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            msgs = conv['messages']
            c_id = conv['conversation_id']
            
            for i in range(len(msgs) - 1):
                if msgs[i]['role'] == 'customer' and msgs[i+1]['role'] == 'support':
                    customer_msg = msgs[i]
                    support_msg = msgs[i+1]
                    
                    tweet_id = customer_msg['tweet_id']
                    intent = intent_map.get(tweet_id, "other_unclear")
                    
                    docs.append({
                        "conversation_id": c_id,
                        "tweet_id": tweet_id,
                        "customer_query": customer_msg['text'],
                        "support_response": support_msg['text'],
                        "intent": intent,
                        "timestamp": customer_msg['created_at']
                    })
                    
    with open(out_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
            
    logger.info(f"Built retrieval corpus with {len(docs)} documents.")

if __name__ == "__main__":
    build_retrieval_corpus()
