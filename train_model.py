import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

data = pd.read_csv('fake_job_postings.csv', sep=',')
pd.set_option('display.max_colwidth', None)
columns_to_combine = ['title','company_profile', 'description', 'requirements', 'benefits']
data['text'] = data[columns_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
X = data['text']
y = data['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_counts, y_train)
labels = model.predict(X_test_counts)
cm = confusion_matrix(y_test, labels)
print(classification_report(y_test, labels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraudulent"])
disp.plot()
plt.show()