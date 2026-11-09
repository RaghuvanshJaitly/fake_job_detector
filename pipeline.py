from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline(classifier):
    pipeline_steps = [('TF-IDF Vectorizer', TfidfVectorizer()),
                      ('Classifier', classifier)]
    
    model_pipeline = Pipeline(pipeline_steps)
    return model_pipeline