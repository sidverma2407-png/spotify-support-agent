import streamlit as st
import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup page
st.set_page_config(page_title="M6 Gold Annotator", layout="wide")

INTENTS = [
    "account_login", "billing_subscription", "playback_app_issue", 
    "content_availability", "ads_recommendations", "cancellation_refund", 
    "agent_handoff", "praise_gratitude", "other_unclear"
]

@st.cache_data
def load_candidates():
    with open('data/gold/gold_candidates.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_annotations():
    csv_path = Path('data/gold/gold_eval.csv')
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    # Return as dict {conversation_id: row_dict}
    return df.set_index('conversation_id').to_dict('index')

def save_annotation(doc, intent, auto_handle, quality, notes):
    csv_path = Path('data/gold/gold_eval.csv')
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=[
        "example_id", "conversation_id", "customer_message", 
        "gold_intent", "should_auto_handle", "response_quality_target", 
        "annotator_notes", "annotator", "timestamp"
    ])
    
    cid = doc['conversation_id']
    row = {
        "example_id": cid,
        "conversation_id": cid,
        "customer_message": doc['customer_query'],
        "gold_intent": intent,
        "should_auto_handle": auto_handle,
        "response_quality_target": quality,
        "annotator_notes": notes if notes else "No notes",
        "annotator": "human_1",
        "timestamp": datetime.now().isoformat()
    }
    
    # Update if exists, else append
    if cid in df['conversation_id'].astype(str).values:
        idx = df.index[df['conversation_id'].astype(str) == cid].tolist()[0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
    df.to_csv(csv_path, index=False)

# Sidebar Guide
with st.sidebar:
    st.header("Annotation Guide")
    st.markdown("""
    **Intents:**
    - `account_login`: Login, hacked, email change.
    - `billing_subscription`: Premium plans, payments, charges.
    - `playback_app_issue`: Bugs, crashes, web-player issues.
    - `content_availability`: Missing songs, regional issues.
    - `ads_recommendations`: Ads complaints, bad recs.
    - `cancellation_refund`: Refunds, unwanted renewals.
    - `agent_handoff`: Explicit requests for human/DM.
    - `praise_gratitude`: Thanks, positive messages.
    - `other_unclear`: Cannot reliably assign other intent.
    """)

# Main execution
candidates = load_candidates()
annotations = load_annotations()

total = len(candidates)
labeled_count = len(annotations)

if 'current_idx' not in st.session_state:
    # Find first unannotated
    idx = 0
    for i, c in enumerate(candidates):
        if str(c['conversation_id']) not in annotations:
            idx = i
            break
    st.session_state.current_idx = idx

current_idx = st.session_state.current_idx

st.title("SpotifyCares Gold Annotation")
st.progress(labeled_count / total)
st.write(f"**Progress: {labeled_count} / {total} labelled**")

if current_idx >= total:
    st.success("All 150 examples labelled! You can close this app.")
    st.stop()

doc = candidates[current_idx]
cid = str(doc['conversation_id'])
existing = annotations.get(cid, {})

st.markdown("---")
st.subheader(f"Example ID: {cid}")
st.info(f"**MODEL PREDICTION (WEAK) — NOT GOLD LABEL:** `{doc['intent']}`")
st.code(doc['customer_query'], language="text")
st.markdown("---")

# Form
with st.form("annotation_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        default_intent = INTENTS.index(existing.get('gold_intent')) if existing.get('gold_intent') in INTENTS else 0
        intent = st.selectbox("1. Gold Intent", INTENTS, index=default_intent)
        
        default_auto = 0 if existing.get('should_auto_handle') == 'yes' else 1
        auto_handle = st.radio("2. Should auto-handle?", ['yes', 'no'], index=default_auto, horizontal=True)
        
    with col2:
        qualities = ['good', 'acceptable', 'poor', 'not_applicable']
        default_qual = qualities.index(existing.get('response_quality_target')) if existing.get('response_quality_target') in qualities else 0
        quality = st.selectbox("3. Response Quality Target", qualities, index=default_qual)
        
        default_notes = existing.get('annotator_notes', '')
        if default_notes == "No notes": default_notes = ""
        notes = st.text_input("4. Optional Notes", value=default_notes)
        
    cols = st.columns([1, 1, 4])
    with cols[0]:
        submitted = st.form_submit_button("Save & Next", type="primary")
    with cols[1]:
        prev = st.form_submit_button("Previous")

if submitted:
    save_annotation(doc, intent, auto_handle, quality, notes)
    st.session_state.current_idx += 1
    st.rerun()

if prev and current_idx > 0:
    st.session_state.current_idx -= 1
    st.rerun()
