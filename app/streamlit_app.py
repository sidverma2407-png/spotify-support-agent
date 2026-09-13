import streamlit as st
import json
import time
from collections import Counter
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.intent.data_loader import load_and_split_data
from src.retrieval.evaluate_retrieval import get_train_test_split
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.generation.generator import ResponseGenerator

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SpotifyCares Agent Demo",
    page_icon="🎧",
    layout="wide"
)

# --- 2. PIPELINE INITIALIZATION (Cached) ---
@st.cache_resource(show_spinner="Initializing AI Support Pipeline...")
def load_pipeline():
    # 1. Load Data
    train_ex, _ = load_and_split_data(random_state=42)
    X_train = [ex['text'] for ex in train_ex]
    y_train = [ex['weak_intent'] for ex in train_ex]
    
    # 2. Train Intent Classifier (FeatureUnion Pipeline from M8)
    clf_pipeline = Pipeline([
        ('feats', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=25000, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=25000, sublinear_tf=True))
        ])),
        ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-4, l1_ratio=0.15, random_state=42, max_iter=1500, class_weight='balanced'))
    ])
    clf_pipeline.fit(X_train, y_train)
    
    # 3. Load Retrieval Corpus
    corpus = []
    with open("data/processed/retrieval_corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
            
    train_corpus, _ = get_train_test_split(corpus, random_state=42)
    retriever = TfidfRetriever(train_corpus)
    
    # 4. Initialize Generator
    generator = ResponseGenerator()
    
    return clf_pipeline, retriever, generator

clf_pipeline, retriever, generator = load_pipeline()

# --- 3. HELPER FUNCTIONS ---
def process_message(msg: str):
    # 1. Intent Classification
    pred_probs = clf_pipeline.predict_proba([msg])[0]
    pred_intent_idx = pred_probs.argmax()
    pred_intent = clf_pipeline.classes_[pred_intent_idx]
    confidence = float(pred_probs[pred_intent_idx])
    
    # 2. Retrieval
    retrieved_examples = retriever.search(msg, top_k=5)
    
    # 3. Generation and Escalation Policy
    result = generator.generate_response(msg, pred_intent, retrieved_examples, confidence)
    
    return result, retrieved_examples

# --- 4. UI LAYOUT ---
st.title("🎧 SpotifyCares Support Agent")
st.markdown("### AI-powered customer support grounded in historical interactions.")

# System Flow Diagram
with st.expander("ℹ️ System Flow Architecture", expanded=False):
    st.markdown("""
    **Customer Message** ➔ **Intent Classification** (Ensemble TF-IDF) ➔ 
    **Historical Retrieval** (Vector Search) ➔ **Grounded Response** ➔ **Policy Decision** (Auto-handle/Escalate)
    """)

# Main columns
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Customer Input")
    
    # Example selection
    examples = {
        "Custom Message": "",
        "Example 1 (Content): Missing Song": "@SpotifyCares why did you remove the new Taylor Swift album? I can't find it anymore.",
        "Example 2 (Billing): Double Charge": "@SpotifyCares I was charged twice for my family plan this month. Please refund.",
        "Example 3 (Playback): App Crashing": "@SpotifyCares the android app keeps crashing whenever I try to play offline songs.",
        "Example 4 (Hacked): Unauthorized Email Change": "@SpotifyCares my email address got changed NOT by me. someone used a throwaway email!"
    }
    
    selected_example = st.selectbox("Choose an example or write your own:", list(examples.keys()))
    
    default_text = examples[selected_example]
    user_input = st.text_area("Customer Message", value=default_text, height=150)
    
    analyze_btn = st.button("Analyze & Generate Response", type="primary")

with right_col:
    st.subheader("Agent Output")
    
    if analyze_btn and user_input:
        with st.spinner("Processing request..."):
            time.sleep(0.5) # Slight delay for UX
            result, retrieved_evidence = process_message(user_input)
            
            # 1. Intent & Confidence
            st.markdown(f"**Predicted Intent:** `{result['intent']}`")
            st.progress(result['confidence'], text=f"Confidence: {result['confidence']:.1%}")
            
            # 2. Decision
            if result['action'] == 'auto_handle':
                st.success(f"**Decision: Auto-handle**")
            else:
                st.warning(f"**Decision: Escalate**")
                st.markdown(f"*Reason:* {result.get('reason', 'Safety/Policy Escalation')}")
                
            # 3. Response
            st.markdown("### Suggested Response")
            st.info(result['response'])
            
            # 4. Historical Evidence
            st.markdown("### Grounded Evidence")
            if retrieved_evidence:
                top_evidence = retrieved_evidence[0]
                st.markdown(f"**Historically Similar Query (Score: {top_evidence['score']:.2f}):**")
                st.markdown(f"> *{top_evidence['customer_query']}*")
                st.markdown(f"**Historical Agent Response:**")
                st.markdown(f"> {top_evidence['support_response']}")
            else:
                st.markdown("*No historical evidence retrieved.*")
    elif analyze_btn:
        st.error("Please enter a customer message.")
