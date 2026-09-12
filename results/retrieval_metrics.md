# Retrieval Evaluation

## Weak-Label Intent Agreement
*(Note: This metric measures whether the retrieved historical conversation shares the same weak intent label as the test query. It is a diagnostic proxy, NOT a true human relevance judgment.)*
| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.487 | 0.743 | 0.839 | 0.919 | 0.633 |
| Embedding | 0.552 | 0.776 | 0.848 | 0.912 | 0.678 |

## Exact Historical Conversation Retrieval
*(Note: Expected to be 0.0 because of strict conversation-level train/test isolation, preventing self-retrieval)*
| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Embedding | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

