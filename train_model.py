import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

#Get data
data = pd.read_csv('fake_job_postings.csv', sep=',')
pd.set_option('display.max_colwidth', None)
columns_to_combine = ['title','company_profile', 'description', 'requirements', 'benefits']
data['text'] = data[columns_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
data = data.drop_duplicates(subset='text', keep='first')
X = data['text']
y = data['fraudulent']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
X_train_counts_tf = tfidf_vectorizer.fit_transform(X_train)
X_test_counts_tf = tfidf_vectorizer.transform(X_test)
#initialize model
model = MultinomialNB()
model2 = MultinomialNB(alpha=0.1)
model.fit(X_train_counts, y_train)
labels = model.predict(X_test_counts)
model2.fit(X_train_counts_tf, y_train)
labels2 = model2.predict(X_test_counts_tf)

#confusion matrix
cm = confusion_matrix(y_test, labels)
cm2 = confusion_matrix(y_test, labels2)
print(classification_report(y_test, labels))
print(classification_report(y_test, labels2))
#plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraudulent"])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Legitimate", "Fraudulent"])
disp.plot()
disp2.plot()
plt.show()