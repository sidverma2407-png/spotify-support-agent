import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def reconstruct_conversations(df: pd.DataFrame, target_author: str = "SpotifyCares") -> List[Dict]:
    logger.info(f"Reconstructing conversations for {target_author}")
    
    # 1. Identify all tweets authored by target_author
    support_tweets = df[df['author_id'] == target_author]
    logger.info(f"Found {len(support_tweets)} tweets authored by {target_author}")
    
    # 1. Identify all tweets authored by target_author
    support_tweets = df[df['author_id'] == target_author]
    logger.info(f"Found {len(support_tweets)} tweets authored by {target_author}")
    
    # Fast single-pass graph extraction
    logger.info("Building parent-child relationships in a single pass...")
    parent_map = {}
    child_map = {}
    
    for t_id, r_id in zip(df['tweet_id'], df['in_response_to_tweet_id']):
        t_id = str(t_id)
        r_id = str(r_id)
        if r_id and r_id != 'nan' and r_id != 'None':
            parent_map[t_id] = r_id
            if r_id not in child_map:
                child_map[r_id] = []
            child_map[r_id].append(t_id)
            
    logger.info("Extracting connected components for target author...")
    relevant_ids = set(support_tweets['tweet_id'].astype(str))
    
    queue = list(relevant_ids)
    visited = set(relevant_ids)
    
    while queue:
        curr = queue.pop(0)
        
        # Add parent
        if curr in parent_map:
            p = parent_map[curr]
            if p not in visited:
                visited.add(p)
                queue.append(p)
                
        # Add children
        if curr in child_map:
            for c in child_map[curr]:
                if c not in visited:
                    visited.add(c)
                    queue.append(c)
                    
    relevant_ids = visited
    logger.info(f"Subgraph extraction complete. {len(relevant_ids)} tweets found.")
    
    # Filter dataframe to only relevant tweets
    sub_df = df[df['tweet_id'].isin(relevant_ids)]
    
    # Now build the dictionary on the much smaller subset!
    logger.info("Building tweet dictionary on subset...")
    tweet_dict = {str(row.tweet_id): row for row in sub_df.itertuples(index=False)}
    
    logger.info("Tracing roots...")
    roots = set()
    for row in sub_df[sub_df['author_id'] == target_author].itertuples(index=False):
        current_id = str(row.tweet_id)
        current_row = row
        visited = set()
        
        while hasattr(current_row, 'in_response_to_tweet_id') and str(current_row.in_response_to_tweet_id) != "" and str(current_row.in_response_to_tweet_id) != "nan":
            parent_id = str(current_row.in_response_to_tweet_id)
            if parent_id in visited or parent_id not in tweet_dict:
                break
            visited.add(parent_id)
            current_id = parent_id
            current_row = tweet_dict[parent_id]
            
        roots.add(current_id)
        
    logger.info(f"Found {len(roots)} conversation roots")
    
    logger.info("Building children map...")
    children_map = {}
    for tid, row in tweet_dict.items():
        parent_id = str(row.in_response_to_tweet_id) if pd.notna(row.in_response_to_tweet_id) else ""
        if parent_id and parent_id != "nan":
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(tid)
            
    conversations = []
    
    logger.info("Reconstructing trees...")
    for root_id in roots:
        messages = []
        queue = [root_id]
        tree_nodes = set()
        
        while queue:
            curr = queue.pop(0)
            if curr in tree_nodes: continue
            tree_nodes.add(curr)
            if curr in children_map:
                queue.extend(children_map[curr])
                
        for node_id in tree_nodes:
            if node_id not in tweet_dict: continue
            row = tweet_dict[node_id]
            role = "support" if str(row.author_id) == target_author else "customer"
            messages.append({
                "tweet_id": node_id,
                "role": role,
                "text": str(row.cleaned_text),
                "original_text": str(row.text),
                "created_at": str(row.created_at)
            })
            
        messages.sort(key=lambda x: x['created_at'])
        
        conversations.append({
            "conversation_id": root_id,
            "messages": messages
        })
        
    return conversations

def main():
    logging.basicConfig(level=logging.INFO)
    from src.data.preprocess import preprocess_dataset
    
    raw_path = "data/raw/twcs.csv"
    if not Path(raw_path).exists():
        logger.error(f"Raw data not found at {raw_path}")
        return
        
    logger.info("Loading raw data...")
    df = pd.read_csv(raw_path)
    
    logger.info("Preprocessing data...")
    df_clean = preprocess_dataset(df)
    
    conversations = reconstruct_conversations(df_clean, target_author="SpotifyCares")
    
    out_path = Path("data/processed/spotify_conversations.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(conversations)} conversations to {out_path}")
    with open(out_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
            
if __name__ == "__main__":
    main()
