import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def clean_tweet_text(text: str) -> str:
    """
    Conservatively clean tweet text:
    - Remove URLs
    - Normalize whitespace
    - Preserve original punctuation, negations, and emojis
    """
    if not isinstance(text, str):
        return ""
        
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw customer support dataset.
    """
    logger.info(f"Initial shape: {df.shape}")
    
    # Normalize columns to lowercase just in case
    df.columns = [c.lower() for c in df.columns]
    
    required_cols = ['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'response_tweet_id', 'in_response_to_tweet_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")
            
    # Remove exact duplicate tweet_ids
    df = df.drop_duplicates(subset=['tweet_id'])
    logger.info(f"Shape after duplicate tweet_id removal: {df.shape}")
    
    # Parse created_at safely
    # Dates are like: Tue Oct 31 22:10:47 +0000 2017
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
    
    # Handle missing text
    df['text'] = df['text'].fillna("")
    
    # Create cleaned_text (vectorized for speed)
    logger.info("Applying text cleaning to create 'cleaned_text'...")
    df['cleaned_text'] = df['text'].str.replace(r'http\S+|www\.\S+', '', regex=True)
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Convert ID columns to strings, treating NaN as empty string
    for col in ['tweet_id', 'author_id', 'response_tweet_id', 'in_response_to_tweet_id']:
        df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)
        df[col] = df[col].replace('nan', '')
    
    # Ensure inbound is boolean
    df['inbound'] = df['inbound'].astype(bool)
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("This module provides preprocessing functions. Run the pipeline via other scripts.")
