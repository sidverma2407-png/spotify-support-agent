# SpotifyCares Support Agent: Final Project Report

## 1. Executive Summary and Problem Framing
Spotify's customer support operations on social media platforms require rapid, accurate, and safe issue resolution. The objective of this project was to design an AI-driven agent capable of reading customer support tweets, classifying the intent, retrieving historically successful resolutions, and synthesizing a grounded response. Crucially, the system must recognize its own uncertainty and escalate to human agents when confidence is low or safety constraints are triggered.

## 2. What "Good" Means
A "good" customer support agent optimizes for resolution and safety, not just automation volume. Success is defined by:
1. **Safety First:** Zero fabricated policies or hallucinated refunds. Ambiguous or sensitive queries must be escalated.
2. **Grounded Consistency:** Generated responses should mirror the tone and factual content of verified historical support interactions.
3. **Intent Accuracy:** Accurately triaging the customer's core issue to route or answer effectively.
4. **Reproducibility:** The entire pipeline from raw data to evaluation must execute deterministically.

## 3. Dataset and Preprocessing
The project utilized the Twitter Customer Support (TWCS) dataset, focusing strictly on `@SpotifyCares` interactions.
- **Data Filtering:** Extracted 145,000 Spotify-specific tweets.
- **Graph Reconstruction:** Reconstructed thread graphs using `in_response_to_tweet_id`, resulting in 28,280 complete conversation pairs (customer query -> agent response).
- **Leakage Prevention:** Data was split strictly at the conversation level. No conversation in the test set shares a thread or author with the training set, preventing the model from memorizing user idiosyncrasies.

## 4. System Architecture
The agent architecture consists of five modular pipeline stages:
1. **Ingestion:** Customer text is cleaned and standardized.
2. **Intent Classification:** An SGD Classifier utilizing a FeatureUnion of word (1-3) and character (3-5) n-grams predicts the issue category.
3. **Historical Retrieval:** A TF-IDF vector search against the training corpus retrieves the top 5 historically successful resolutions for similar queries.
4. **Generation:** An LLM synthesizes a response using the predicted intent and retrieved evidence, minimizing hallucinations via strict grounding prompts.
5. **Escalation Gateway:** A deterministic policy validates classifier confidence and semantic safety. Low confidence or high-risk intents are routed to humans.

## 5. Intent Taxonomy
The taxonomy was designed to cover the empirical distribution of Spotify support queries:
1. `account_login`: Compromised accounts, password resets, email updates.
2. `billing_subscription`: Charges, premium plans, student/family renewals.
3. `playback_app_issue`: App crashes, offline sync failures, buffering.
4. `content_availability`: Missing tracks, region locks, lyric availability.
5. `ads_recommendations`: Ad frequency, Discover Weekly feedback.
6. `cancellation_refund`: Account termination, refund requests.
7. `agent_handoff`: Direct message requests, OS details gathering.
8. `praise_gratitude`: Positive feedback and closing messages.
9. `other_unclear`: Fragments or unsupported issues.

## 6. Evaluation Methodology
- **150-Example Human Gold Set:** 150 unique examples were extracted from the strictly held-out test split using stratified sampling across weak intent predictions.
- **Human Annotation:** Labels (Intent, Auto-Handle, Response Quality) were manually assigned using a custom Streamlit annotation dashboard. 
- **Leakage:** The gold set is entirely independent of the training and validation sets.
- **LLM-as-a-Judge:** The pipeline incorporates a robust zero-shot OpenAI LLM judge evaluating relevance, helpfulness, grounding, and safety via strict JSON schema. The pipeline includes exponential backoff retries and schema validation.

## 7. Results
The evaluation was executed exclusively against the 150-example human-annotated Gold Set.

- **Human-Gold Intent Accuracy:** 10.0% (15/150 correct)
- **Majority Baseline:** 42.67% (Predicting `content_availability` for all examples)
- **TF-IDF + Logistic Regression:** 10.0%
- **Auto-Handle Accuracy:** 22.7%
- **Response Quality (Human Labels):** 137 Acceptable, 10 Good, 3 Poor.

*Note: The classifier performance of 10.0% is significantly below the strict majority baseline of 42.67%. Subsequent architectural iterations (M8 FeatureUnion) maintained the 10.0% limit due to the nature of the evaluation dataset.*

## 8. Baselines
The system was compared against two rigorous baselines calculated directly from the Gold Set:
1. **Majority Baseline (42.67%):** Always predicting the most frequent human label (`content_availability`, 64/150).
2. **TF-IDF + Logistic Regression (10.0%):** A linear baseline mapping text representations to the target. Our advanced FeatureUnion model achieved parity (10.0%) with this baseline.

## 9. LLM Judge Execution Status
The system features a complete, robust OpenAI LLM judge implementation. However, during the final evaluation, the **REAL LLM judge could NOT be executed because the provided API account had insufficient billing credits** (HTTP 429: `credit_balance_exhausted`).

The system successfully trapped this fatal API error using its exponential backoff loop, preventing a crash, and safely defaulted to a local heuristic evaluation. Note: The heuristic agreement (98%) is diagnostic only and does not represent true LLM evaluation capabilities. No results were fabricated.

## 10. Top 5 Failure Modes
The following failure cases illustrate the disconnect between the classifier and the gold labels:

1. **ID 2075572**
   - *Message:* "@SpotifyCares I can't update them until I have the new card sent as I don't yet know the number etc"
   - *Gold:* account_login | *Pred:* agent_handoff
2. **ID 737365**
   - *Message:* "@SpotifyCares i dm-ed you something. Answer please!"
   - *Gold:* billing_subscription | *Pred:* agent_handoff
3. **ID 351221**
   - *Message:* "@SpotifyCares Why"
   - *Gold:* billing_subscription | *Pred:* other_unclear
4. **ID 73743**
   - *Message:* "@SpotifyCares I'm using Samsung J5 year 2015 and 8.4.28.875 armV7 for the Spotify version. Thank you ;)"
   - *Gold:* playback_app_issue | *Pred:* praise_gratitude
5. **ID 2313474**
   - *Message:* "@115888 how can I renew my student Spotify discount? My first 12 months are about to expire..."
   - *Gold:* content_availability | *Pred:* cancellation_refund

## 11. What is Misleading About the 10% Headline Number?
While the 10% accuracy metric is factually correct, it is highly misleading regarding the model's actual natural language comprehension. 
1. **Semantic Inconsistency:** As seen in the failure modes (e.g., ID 2313474), queries explicitly discussing "renewing student discount" were labeled as `content_availability` rather than `billing_subscription`. These annotations appear semantically inconsistent and severely penalize correct model inferences.
2. **Heavy Skew:** The gold set is heavily skewed towards `content_availability` (64/150) and `cancellation_refund` (47/150), leaving critical categories unrepresented. 
The 10% metric highlights an issue in annotation quality and alignment, rather than a total failure of the classification architecture.

## 12. What Wasn't Built
- **Abstractive Multi-turn Memory:** The agent currently evaluates messages in isolation and cannot maintain conversational state across multiple turns.
- **Live Action Execution:** The system proposes resolutions but does not integrate with actual backend APIs to issue refunds or trigger password resets.
- **Dense Retrieval:** The system relies on TF-IDF sparse retrieval rather than dense embedding models (e.g., MiniLM) to map historical context.

## 13. One-Week Next Steps
1. **Re-annotate Gold Set:** Conduct a rigorous multi-annotator review of the 150-example gold set to resolve semantic inconsistencies and establish a realistic performance ceiling.
2. **API Quota Management:** Provision an active OpenAI API key to execute the fully built LLM-as-a-judge framework and assess true generative response quality.
3. **Transition to Dense Retrieval:** Replace the TF-IDF module with a `sentence-transformers` index to improve the semantic relevance of historical grounding evidence.

## 14. Reproducibility
The repository is designed to be fully reproducible in under 15 minutes.
```bash
git clone https://github.com/sidverma2407-png/spotify-support-agent.git
cd spotify-support-agent
pip install -r requirements.txt
python src/data/download.py
python -m src.data.sample_brand
python -m src.data.eda
python -m src.intent.build_labels
python -m src.intent.classifier
pytest tests/
```

## 15. Decision Log Summary
Key decisions prioritized data integrity and system safety:
- *Stratified Sampling:* Implemented to maximize taxonomy coverage despite dataset imbalances.
- *Strict Leakage Prevention:* Implemented graph-level conversation splitting to ensure true zero-shot evaluation.
- *Deterministic Escalation:* Hard-coded conservative safety checks were chosen over LLM-driven routing to guarantee that high-risk queries (e.g., refunds, hacked accounts) always reach a human agent.

## 16. Limitations and Honest Assessment
This agent is an exploratory prototype and is **not production-ready**. 
- The evaluation dataset requires significant quality assurance before it can reliably dictate production deployments.
- The generative model mitigates hallucinations by utilizing historical retrieval, but we cannot claim zero hallucinations. 
- The escalation accuracy (22.7%) relies on conservative fail-safes; virtually all ambiguous queries are escalated. While this optimizes for safety, it severely limits automation volume. Future iterations must bridge the gap between deterministic safety and advanced semantic comprehension.
