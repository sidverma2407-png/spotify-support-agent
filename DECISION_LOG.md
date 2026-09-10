# Architecture & Engineering Decision Log

This document records the key architectural choices, trade-offs, and engineering principles adopted during the development of the **Spotify Support Agent** for the Hiver SDE Intern take-home assignment.

---

## ADR-001: Project Foundation & Modular Pipeline Architecture

* **Date:** 2026-09-10
* **Status:** Accepted
* **Milestone:** M1 (Project Foundation)

### Context & Problem Statement
Customer support systems require high reliability, predictable latency, and verifiable factual correctness. In an enterprise setting (such as Hiver's shared inbox and customer communication platform), hallucinated or generic canned responses harm user trust. The challenge is to construct an evidence-grounded AI support agent specifically for `@SpotifyCares` that is modular, testable, and explainable in an engineering interview.

### Decisions

#### 1. Modular Decoupled Pipeline
We decoupled the system into five explicit functional modules under `src/`:
* `src/data/`: Manages raw data parsing, thread reconstruction, and train/val/test splits.
* `src/intent/`: Classifies user issues into actionable support categories (e.g., billing, audio playback, account recovery).
* `src/retrieval/`: Finds grounded historical resolutions and knowledge base articles matching the user query.
* `src/generation/`: Generates clear, brand-aligned answers conditioned strictly on retrieved evidence.
* `src/escalation/`: Implements risk thresholds and confidence scores to route ambiguous, sensitive, or unresolved cases to human agents.

*Rationale*: A modular architecture allows each component to be unit-tested, benchmarked, and iterated upon independently without requiring changes to the rest of the system.

#### 2. Tiered Data Management (`raw/`, `processed/`, `gold/`)
We established strict directory boundaries:
* `data/raw/`: Read-only storage for original datasets (e.g., Customer Support on Twitter).
* `data/processed/`: Filtered and cleaned `@SpotifyCares` conversations.
* `data/gold/`: Immutable, curated evaluation benchmark dataset.

*Rationale*: Strict separation prevents data leakage between model training and performance reporting, ensuring that evaluation numbers are genuine and reproducible.

#### 3. Evaluation-First Mindset over Feature Creep
Instead of jumping directly into complex multi-agent frameworks or LLM wrappers, the project starts with clean boundaries, verifiable baselines, and reproducible test fixtures.

*Rationale*: In a production engineering assessment, clean architecture, failure handling, and rigorous evaluation metrics are vastly more valuable than superficial features.

#### 4. Lean Dependency Strategy
Initial dependencies in `requirements.txt` are constrained to foundational tools (`pandas`, `numpy`, `scikit-learn`, `pytest`, `python-dotenv`). Additional specialized libraries will only be added as needed in respective milestones.

*Rationale*: Minimizes dependency conflicts, keeps local setup fast, and ensures code portability across environments.

---

## ADR-002: Data Processing & Conversation Reconstruction (M2)

* **Date:** 2026-09-10
* **Status:** Accepted
* **Milestone:** M2 (Data Processing & EDA)

### Context
To model SpotifyCares responses, we must extract clean, coherent conversation threads from a 2.8 million-row Customer Support on Twitter dataset. The data contains fragmented replies, missing text, and noisy text formatting.

### Decisions

#### 1. Conservative Text Cleaning
Text cleaning specifically limits itself to:
- Stripping explicit URLs (`http://...`).
- Normalizing redundant whitespaces.
- Preserving all native emojis, negations, capitalization, and punctuation.

*Rationale*: Deep learning language models and NLP evaluators heavily rely on punctuation (like question marks) and emojis (sentiment indicators). Stripping them blindly harms the signal.

#### 2. Reconstructing Conversations from Tweet IDs
Instead of treating tweets as isolated documents, we construct trees using the `in_response_to_tweet_id` pointer. A DFS/BFS approach groups messages into `{"role": "...", "text": "..."}` turns starting from the root customer query.

*Rationale*: LLMs need the full multi-turn context (e.g., "Customer: my app crashed" -> "Agent: try this" -> "Customer: didn't work") to generate accurate follow-ups.

#### 3. No Target Data Leakage
The raw dataset is extracted via Kaggle API (`kagglehub`) directly to `data/raw/` and explicitly `.gitignore`'d.

*Rationale*: Ensures the repository remains lean and forces developers to build reproducible pipelines rather than relying on manual file edits.
