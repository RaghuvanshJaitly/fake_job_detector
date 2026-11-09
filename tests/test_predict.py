import numpy as np
from predict import predict_postings



class FakePostingModel:
    def predict_proba(self, postings):
        return np.array([
    [0.91, 0.09],
    [0.90, 0.10],
    [0.89, 0.11],
    ])
        
def test_postings_applies_threshold_to_batch():
    postings = ["PostingA", "PostingB", "PostingC"]
    loaded_model = {"pipeline": FakePostingModel(), "threshold": 0.10}

    prob, decision = predict_postings(postings, loaded_model)

    assert prob.tolist() == [0.09, 0.10, 0.11]
    assert decision.tolist() == [False, True, True]