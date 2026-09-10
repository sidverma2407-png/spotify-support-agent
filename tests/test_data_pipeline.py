import pandas as pd
import pytest
from src.data.preprocess import clean_tweet_text, preprocess_dataset
from src.data.sample_brand import reconstruct_conversations

def test_text_cleaning():
    # URL removal
    assert clean_tweet_text("Check this out http://example.com") == "Check this out"
    # Normalizing whitespace
    assert clean_tweet_text("Too   much \n whitespace") == "Too much whitespace"
    # Preserving punctuation, negations, emojis
    text_with_emoji = "I don't like this! 😭"
    assert clean_tweet_text(text_with_emoji) == text_with_emoji
    # Empty/NaN safe
    assert clean_tweet_text(None) == ""

def test_preprocess_dataset():
    data = {
        'tweet_id': ['1', '2', '2'], # 2 is duplicate
        'author_id': ['SpotifyCares', 'customer1', 'customer1'],
        'inbound': ['False', 'True', 'True'],
        'created_at': ['2017-10-31 22:21:40', '2017-10-31 22:21:40', 'bad_date'],
        'text': ['Hello http://url.com', None, 'duplicate'],
        'response_tweet_id': ['2', '', ''],
        'in_response_to_tweet_id': ['', '1', '1']
    }
    df = pd.DataFrame(data)
    
    clean_df = preprocess_dataset(df)
    
    # Check duplicate removal
    assert len(clean_df) == 2
    
    # Check null text handling
    customer_row = clean_df[clean_df['tweet_id'] == '2'].iloc[0]
    assert customer_row['text'] == ""
    
    # Check string type for IDs
    assert clean_df['tweet_id'].dtype in [object, 'O', 'string', 'str']
    assert clean_df['author_id'].dtype in [object, 'O', 'string', 'str']
    
    # Check inbound boolean
    assert customer_row['inbound'] == True

def test_reconstruct_conversations():
    # Toy conversation:
    # customer (t1) -> support (t2) -> customer (t3)
    data = {
        'tweet_id': ['1', '2', '3', '4'],
        'author_id': ['customer1', 'SpotifyCares', 'customer1', 'other_brand'],
        'inbound': [True, False, True, False],
        'created_at': ['2017-10-31 10:00:00', '2017-10-31 10:05:00', '2017-10-31 10:10:00', '2017-10-31 10:00:00'],
        'text': ['Help!', 'How can we help?', 'It broke', 'Other issue'],
        'cleaned_text': ['Help!', 'How can we help?', 'It broke', 'Other issue'],
        'response_tweet_id': ['2', '3', '', ''],
        'in_response_to_tweet_id': ['', '1', '2', '']
    }
    df = pd.DataFrame(data)
    
    # convert dates for sort to work normally (or leave as string, simple sort works on ISO)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    conversations = reconstruct_conversations(df, target_author='SpotifyCares')
    
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv['conversation_id'] == '1' # root tweet
    
    msgs = conv['messages']
    assert len(msgs) == 3
    
    assert msgs[0]['tweet_id'] == '1'
    assert msgs[0]['role'] == 'customer'
    
    assert msgs[1]['tweet_id'] == '2'
    assert msgs[1]['role'] == 'support'
    
    assert msgs[2]['tweet_id'] == '3'
    assert msgs[2]['role'] == 'customer'
