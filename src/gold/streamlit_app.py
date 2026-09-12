import streamlit as st
import json
import csv
import pandas as pd
from pathlib import Path
import random

st.set_page_config(page_title="M6 Gold Annotator", layout="wide")

INTENTS = [
    "--- Select Intent ---",
    "account_login", "billing_subscription", "playback_app_issue", 
    "content_availability", "ads_recommendations", "cancellation_refund", 
    "agent_handoff", "praise_gratitude", "other_unclear"
]
QUALITIES = ["--- Select Quality ---", "good", "acceptable", "poor", "not_applicable"]
AUTO_HANDLE = ["--- Select ---", "yes", "no"]

@st.cache_data
def load_candidates():
    with open('data/gold/gold_candidates.json', 'r', encoding='utf-8') as f:
        cands = json.load(f)
        # Randomize order using fixed seed
        random.Random(42).shuffle(cands)
        return cands

def load_annotations():
    csv_path = Path('data/gold/gold_eval.csv')
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    return df.set_index('conversation_id').to_dict('index')

def save_annotation(doc, intent, auto_handle, quality, notes):
    csv_path = Path('data/gold/gold_eval.csv')
    df = pd.read_csv(csv_path, dtype=str) if csv_path.exists() else pd.DataFrame(columns=[
        "example_id", "conversation_id", "customer_message", 
        "gold_intent", "should_auto_handle", "response_quality_target", "annotator_notes"
    ])
    
    cid = str(doc['conversation_id'])
    row = {
        "example_id": cid,
        "conversation_id": cid,
        "customer_message": doc['customer_query'],
        "gold_intent": intent,
        "should_auto_handle": auto_handle,
        "response_quality_target": quality,
        "annotator_notes": notes if notes else "No notes"
    }
    
    if cid in df['conversation_id'].astype(str).values:
        idx = df.index[df['conversation_id'].astype(str) == cid].tolist()[0]
        for k, v in row.items():
            df.at[idx, k] = str(v)
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
    df.to_csv(csv_path, index=False)

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

candidates = load_candidates()
annotations = load_annotations()

total = len(candidates)
human_labeled_count = len(annotations)

if 'current_idx' not in st.session_state:
    idx = 0
    for i, c in enumerate(candidates):
        cid = str(c['conversation_id'])
        if cid not in annotations:
            idx = i
            break
    st.session_state.current_idx = idx

current_idx = st.session_state.current_idx

st.title("SpotifyCares Gold Annotation")
st.progress(min(human_labeled_count / total, 1.0))
st.write(f"**Human reviewed: {human_labeled_count} / {total}**")

st.warning("⚠️ **Do not copy the model prediction. Make an independent judgment.**\n\nA label matching the model prediction is perfectly valid if you independently reach the same conclusion. We should NOT artificially force disagreement.")

if current_idx >= total:
    st.success("All 150 examples genuinely human-reviewed! You can close this app.")
    st.stop()

doc = candidates[current_idx]
cid = str(doc['conversation_id'])
existing = annotations.get(cid, {})

st.markdown("---")
st.subheader(f"Example ID: {cid}")
st.info(f"**MODEL PREDICTION (WEAK) — NOT GOLD LABEL:** `{doc['intent']}`")
st.code(doc['customer_query'], language="text")
st.markdown("---")

with st.form("annotation_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        default_intent = INTENTS.index(existing.get('gold_intent')) if existing.get('gold_intent') in INTENTS else 0
        intent = st.selectbox("1. Gold Intent", INTENTS, index=default_intent)
        
        default_auto = AUTO_HANDLE.index(existing.get('should_auto_handle')) if existing.get('should_auto_handle') in AUTO_HANDLE else 0
        auto_handle = st.radio("2. Should auto-handle?", AUTO_HANDLE, index=default_auto, horizontal=True)
        
    with col2:
        default_qual = QUALITIES.index(existing.get('response_quality_target')) if existing.get('response_quality_target') in QUALITIES else 0
        quality = st.selectbox("3. Response Quality Target", QUALITIES, index=default_qual)
        
        notes = st.text_input("4. Optional Notes", value=existing.get('annotator_notes', '') if existing.get('annotator_notes') != "No notes" else "")
        
    cols = st.columns([1, 1, 4])
    with cols[0]:
        submitted = st.form_submit_button("Save & Next", type="primary")
    with cols[1]:
        prev = st.form_submit_button("Previous")

if submitted:
    if intent.startswith("---") or auto_handle.startswith("---") or quality.startswith("---"):
        st.error("❌ Please explicitly select all required fields before saving.")
    else:
        save_annotation(doc, intent, auto_handle, quality, notes)
        st.session_state.current_idx += 1
        st.rerun()

if prev and current_idx > 0:
    st.session_state.current_idx -= 1
    st.rerun()
