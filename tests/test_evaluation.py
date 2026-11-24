from evaluation import evaluate_at_k

#If I review only the k highest-scoring postings, how much fraud will I catch?
def test_evaluate_at_k_selects_highest_scores():
    scores = [0.40, 0.95, 0.10, 0.80]
    labels = [1, 1, 0, 0]
    k = 2
    result = evaluate_at_k(labels, scores, k)
    assert result["fraud_caught"] == 1
    assert result["false_alarms"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5

    