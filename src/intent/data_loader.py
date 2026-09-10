import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def load_and_split_data(test_size=0.2, random_state=42):
    input_file = Path("data/processed/labeled_intent_data.jsonl")
    
    conv_to_examples = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            conv_to_examples[ex['conversation_id']].append(ex)
            
    # To prevent leakage, split at the conversation ID level
    conversation_ids = list(conv_to_examples.keys())
    
    # We can't strictly stratify by intent across conversations easily 
    # because one conversation can have multiple intents. 
    # But a random split of 28k conversations is naturally well stratified by Law of Large Numbers.
    random.seed(random_state)
    random.shuffle(conversation_ids)
    
    split_idx = int(len(conversation_ids) * (1 - test_size))
    train_convs = set(conversation_ids[:split_idx])
    test_convs = set(conversation_ids[split_idx:])
    
    assert len(train_convs.intersection(test_convs)) == 0, "Leakage detected!"
    
    train_examples = []
    test_examples = []
    
    for cid in train_convs:
        train_examples.extend(conv_to_examples[cid])
        
    for cid in test_convs:
        test_examples.extend(conv_to_examples[cid])
        
    logger.info(f"Train conversations: {len(train_convs)}, Train examples: {len(train_examples)}")
    logger.info(f"Test conversations: {len(test_convs)}, Test examples: {len(test_examples)}")
    
    return train_examples, test_examples
