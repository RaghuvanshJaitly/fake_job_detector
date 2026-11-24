import numpy as np

def evaluate_at_k(y_true, scores, k:int) -> dict:
    labels = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    
    if len(labels) != len(scores):
        raise ValueError("Each score must have a matching label")
    
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and the number of postings.")
    #It gives you the original positions of the postings, ranked from highest score to lowest.
    ranked_indices = np.argsort(-scores, kind="stable")
    #It gets the actual labels of the k highest-scoring postings
    reviewed_labels = labels[ranked_indices[:k]]
    fraud_caught = int(np.sum(reviewed_labels == 1))
    total_fraud = int(np.sum(labels == 1))
    
    return {
        "fraud_caught": fraud_caught,
        "false_alarms": k - fraud_caught,
        "precision": fraud_caught / k,
        "recall": fraud_caught / total_fraud if total_fraud else 0.0,
    }