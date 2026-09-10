# Spotify Support Agent

An evidence-grounded AI customer support agent built on real-world customer support interactions from the *Customer Support on Twitter* dataset (specifically focused on the `@SpotifyCares` brand).

This repository is developed as a production-quality take-home assignment for the **Hiver SDE Intern** role, adhering to engineering rigor, modular pipeline architecture, and an evaluation-first philosophy.

---

## 📌 Architecture & Design Philosophy

Rather than building an opaque monolith, the system is designed as an end-to-end modular pipeline where each stage has clear responsibilities, decoupled interfaces, and measurable baselines:

```text
User Query
    │
    ▼
┌─────────────────────────────────┐
│ 1. Intent Detection             │  Classifies incoming user issues
│    (src/intent)                 │  (e.g., billing, playback, account)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 2. Context Retrieval            │  Retrieves relevant historical resolution
│    (src/retrieval)              │  evidence and grounded troubleshooting steps
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 3. Response Generation          │  Synthesizes helpful, evidence-grounded
│    (src/generation)             │  responses adhering to Spotify brand voice
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 4. Escalation & Triage          │  Detects high-risk, unresolved, or sensitive
│    (src/escalation)             │  queries for human support handover
└─────────────────────────────────┘
```

### Key Engineering Principles
1. **Evidence over Flashiness**: Every architectural and modeling decision is validated against reproducible evaluation metrics on benchmark data rather than subjective impressions.
2. **Modular & Extensible**: Each module in `src/` can be independently tested, improved, or swapped without breaking upstream or downstream components.
3. **Reproducibility**: Clear environment management, versioned data boundaries (`data/raw/`, `data/processed/`, `data/gold/`), and explicit configuration templates (`.env.example`).
4. **Interview Explainability**: Clean, idiomatic Python code with documented trade-offs and decision logs.

---

## 🗂️ Project Structure

```text
spotify-support-agent/
├── data/
│   ├── raw/                 # Unprocessed source data (e.g., twcs.csv)
│   ├── processed/           # Filtered, cleaned @SpotifyCares conversations
│   └── gold/                # Curated benchmark dataset for evaluation
├── src/
│   ├── data/                # Data ingestion, cleaning, and conversation pairing
│   ├── intent/              # Intent classification and routing logic
│   ├── retrieval/           # Evidence retrieval and historical context search
│   ├── generation/          # Grounded response generation
│   ├── escalation/          # Confidence scoring and human-in-the-loop triage
│   └── pipeline.py          # End-to-end execution pipeline entrypoint
├── evaluation/              # Quantitative evaluation scripts and metrics
├── tests/                   # Automated unit and integration test suite
├── notebooks/               # Exploratory data analysis (EDA) and experiments
├── results/                 # Metrics outputs, evaluation tables, and figures
├── app/                     # User interface or API service
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules for data, models, and cache
├── requirements.txt         # Core project dependencies
├── DECISION_LOG.md          # Architecture and engineering decision records
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup Environment
```bash
# Clone the repository (if not already cloned)
git clone https://github.com/sidverma2407-png/spotify-support-agent.git
cd spotify-support-agent

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the sample environment configuration:
```bash
cp .env.example .env
```

### 4. Run the Pipeline
Execute the pipeline entrypoint:
```bash
python src/pipeline.py
```
Expected output:
```text
Spotify Support Agent
Project setup successful!
```

---

## 🗺️ Milestone Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M1: Project Foundation** | Directory layout, dependencies, environment templates, decision log, minimal pipeline entrypoint | ✅ Completed |
| **M2: Data Processing & EDA** | Ingest Kaggle Twitter dataset, filter `@SpotifyCares`, structure multi-turn threads, baseline EDA | ⏳ Pending |
| **M3: Intent Classification** | Categorize user queries into operational buckets (account, billing, audio/app bugs) | ⏳ Pending |
| **M4: Grounded Retrieval** | Index past resolved solutions; retrieve nearest evidence for incoming issues | ⏳ Pending |
| **M5: Response Generation & Tone** | Evidence-grounded response generation with guardrails and Spotify support style | ⏳ Pending |
| **M6: Escalation Logic** | Confidence gating, out-of-scope detection, and human handover routing | ⏳ Pending |
| **M7: Evaluation & Benchmarking** | Formal evaluation metrics (accuracy, retrieval precision, grounding, escalation recall) | ⏳ Pending |
| **M8: App / Demo & Packaging** | Interactive demo interface and final documentation polish | ⏳ Pending |

---

## 📄 Decision Log
All architectural and implementation trade-offs are actively tracked in [`DECISION_LOG.md`](file:///C:/Users/Sid/.gemini/antigravity/scratch/spotify-support-agent/DECISION_LOG.md).
