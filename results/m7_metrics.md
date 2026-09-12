# M7 Evaluation Metrics

## Headline Metric
**Percentage of Examples Correctly Handled (Intent + Appropriate Action)**: 0.0%

*What is misleading about this metric?*
- The gold set is heavily skewed (due to manual user testing artifacts).
- It relies on only 150 examples from a single historical source (Twitter).
- Heuristic fallback judge is used instead of a real LLM.

## Intent Classification
- Accuracy: 0.107 (vs Majority: 0.000)
- Macro F1: 0.021
- Weighted F1: 0.193

## Escalation Policy
- Action Accuracy: 0.040
- System Escalation Rate: 0.960
- Auto-Handle Precision: 1.000 | Recall: 0.040
- Escalate Precision: 0.000 | Recall: 0.000

## Judge vs Human Agreement
- Binary Agreement Accuracy (Pass vs Fail): 1.000
- Note: Evaluated using heuristic deterministic fallback judge because no LLM API is provided.
