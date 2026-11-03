from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def build_pipeline():
    pipeline_steps = [('TF-IDF Vectorizer', TfidfVectorizer()),
                      ('MultinomialNB', MultinomialNB(alpha=0.001))]
    
    model_pipeline = Pipeline(pipeline_steps)
    
    return model_pipeline