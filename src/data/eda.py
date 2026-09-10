import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_eda(df: pd.DataFrame, df_clean: pd.DataFrame, conversations: list, target_author: str = "SpotifyCares") -> Dict[str, Any]:
    # Stats from raw dataset (if needed, but usually we use cleaned for basic stats or both)
    total_tweets = len(df)
    missing_text_count = df['text'].isna().sum()
    duplicate_tweet_ids = total_tweets - df['tweet_id'].nunique()
    
    # Stats from cleaned dataset
    spotify_tweets = df_clean[df_clean['author_id'] == target_author]
    spotify_count = len(spotify_tweets)
    
    inbound_count = df_clean['inbound'].sum()
    outbound_count = len(df_clean) - inbound_count
    
    date_min = df_clean['created_at'].min()
    date_max = df_clean['created_at'].max()
    
    # Stats from reconstructed conversations
    num_conversations = len(conversations)
    conv_lengths = [len(c['messages']) for c in conversations]
    
    # customer -> support pair count
    pair_count = 0
    for conv in conversations:
        msgs = conv['messages']
        # count how many times a customer message is directly followed by a support message
        for i in range(len(msgs) - 1):
            if msgs[i]['role'] == 'customer' and msgs[i+1]['role'] == 'support':
                pair_count += 1
                
    stats = {
        "dataset": {
            "total_raw_tweets": int(total_tweets),
            "duplicate_tweet_ids": int(duplicate_tweet_ids),
            "missing_text_count": int(missing_text_count),
        },
        "cleaned_dataset": {
            "total_tweets": len(df_clean),
            "inbound_customer_tweets": int(inbound_count),
            "outbound_support_tweets": int(outbound_count),
            "target_brand": target_author,
            "target_brand_tweet_count": int(spotify_count),
            "date_range": [str(date_min), str(date_max)]
        },
        "conversations": {
            "num_reconstructed": int(num_conversations),
            "total_customer_support_pairs": int(pair_count),
            "length_distribution": {
                "min": min(conv_lengths) if conv_lengths else 0,
                "max": max(conv_lengths) if conv_lengths else 0,
                "mean": round(sum(conv_lengths)/len(conv_lengths), 2) if conv_lengths else 0
            }
        }
    }
    return stats

def main():
    logging.basicConfig(level=logging.INFO)
    from src.data.preprocess import preprocess_dataset
    
    raw_path = "data/raw/twcs.csv"
    if not Path(raw_path).exists():
        logger.error(f"Raw data not found at {raw_path}")
        return
        
    logger.info("Loading raw data...")
    df = pd.read_csv(raw_path)
    
    logger.info("Preprocessing data for stats...")
    df_clean = preprocess_dataset(df.copy())
    
    conv_path = Path("data/processed/spotify_conversations.jsonl")
    conversations = []
    if conv_path.exists():
        logger.info(f"Loading conversations from {conv_path}...")
        with open(conv_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
    else:
        logger.warning(f"{conv_path} not found. Conversations stats will be empty.")
        
    logger.info("Generating EDA statistics...")
    stats = generate_eda(df, df_clean, conversations)
    
    Path("results").mkdir(exist_ok=True)
    
    with open("results/data_profile.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    md_content = f"""# Data Profile: SpotifyCares

## Dataset Overview
- **Total Raw Tweets**: {stats['dataset']['total_raw_tweets']:,}
- **Duplicates Removed**: {stats['dataset']['duplicate_tweet_ids']:,}
- **Missing Text Handled**: {stats['dataset']['missing_text_count']:,}

## Cleaned Data
- **Total Valid Tweets**: {stats['cleaned_dataset']['total_tweets']:,}
- **Inbound (Customer) Tweets**: {stats['cleaned_dataset']['inbound_customer_tweets']:,}
- **Outbound (Support) Tweets**: {stats['cleaned_dataset']['outbound_support_tweets']:,}
- **Date Range**: {stats['cleaned_dataset']['date_range'][0]} to {stats['cleaned_dataset']['date_range'][1]}

## SpotifyCares Filter
- **Brand Tweets**: {stats['cleaned_dataset']['target_brand_tweet_count']:,}
- **Reconstructed Conversations**: {stats['conversations']['num_reconstructed']:,}
- **Customer -> Support Interaction Pairs**: {stats['conversations']['total_customer_support_pairs']:,}
- **Conversation Length**: 
  - Mean: {stats['conversations']['length_distribution']['mean']} messages
  - Range: {stats['conversations']['length_distribution']['min']} - {stats['conversations']['length_distribution']['max']} messages
"""
    with open("results/data_profile.md", "w") as f:
        f.write(md_content)
        
    logger.info("Saved EDA results to results/data_profile.json and results/data_profile.md")

if __name__ == "__main__":
    main()
