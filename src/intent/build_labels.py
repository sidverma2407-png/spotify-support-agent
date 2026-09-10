import json
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.intent.taxonomy import INTENT_CLASSES

# Heuristic keyword rules
RULES = {
    "cancellation_refund": r'\b(cancel|refund|renew|renewal|money back|unsubscribe)\b',
    "account_login": r'\b(hack|hacked|compromised|email|password|login|log in|sign in|facebook|account|username|forgot)\b',
    "billing_subscription": r'\b(premium|family|student|hulu|charge|pay|bill|subscribe|upgrade|charged|payment|credit card)\b',
    "content_availability": r'\b(missing|where is|region|country|lyrics|available|removed|can\'t find|add)\b',
    "ads_recommendations": r'\b(ad|ads|commercial|discover weekly|suggest|recommendation|recommend|playlist)\b',
    "playback_app_issue": r'\b(play|pause|crash|bug|download|offline|web player|chrome|windows|mac|android|ios|app|update|skip|volume|working|doesn\'t work|error|glitch)\b',
    "praise_gratitude": r'\b(thank|thanks|love|appreciate|sweet|great|awesome|best|fixed|works now)\b',
    "agent_handoff": r'\b(dm|direct message|screenshot|sent|followed|following|done|here is|sent it)\b'
}

def weak_label(text, prev_customer_text=""):
    text_lower = text.lower()
    
    # Check for agent handoff or gratitude first as they are often short terminal nodes
    if re.search(RULES["praise_gratitude"], text_lower):
        return "praise_gratitude"
    if re.search(RULES["agent_handoff"], text_lower):
        return "agent_handoff"
        
    # Check primary issues
    for intent in ["cancellation_refund", "account_login", "billing_subscription", 
                   "content_availability", "ads_recommendations", "playback_app_issue"]:
        if re.search(RULES[intent], text_lower):
            return intent
            
    # If no match in current text, use conversation context (previous customer message)
    if prev_customer_text:
        prev_lower = prev_customer_text.lower()
        for intent in ["cancellation_refund", "account_login", "billing_subscription", 
                       "content_availability", "ads_recommendations", "playback_app_issue"]:
            if re.search(RULES[intent], prev_lower):
                return intent
                
    return "other_unclear"

def build_labels():
    input_file = Path("data/processed/spotify_conversations.jsonl")
    output_file = Path("data/processed/labeled_intent_data.jsonl")
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    labeled_examples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            conv_id = conv['conversation_id']
            
            prev_cust_text = ""
            for msg in conv['messages']:
                if msg['role'] == 'customer':
                    text = msg['text']
                    intent = weak_label(text, prev_cust_text)
                    
                    labeled_examples.append({
                        "conversation_id": conv_id,
                        "tweet_id": msg['tweet_id'],
                        "text": text,
                        "weak_intent": intent
                    })
                    
                    prev_cust_text = text

    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in labeled_examples:
            f.write(json.dumps(ex) + "\n")
            
    logger.info(f"Generated {len(labeled_examples)} weak labels.")
    
    # Print distribution
    dist = {}
    for ex in labeled_examples:
        dist[ex['weak_intent']] = dist.get(ex['weak_intent'], 0) + 1
        
    for k, v in dist.items():
        logger.info(f"  {k}: {v} ({v/len(labeled_examples)*100:.1f}%)")

if __name__ == "__main__":
    build_labels()
