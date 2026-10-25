import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

#Get data
data = pd.read_csv('fake_job_postings.csv', sep=',')
pd.set_option('display.max_colwidth', None)
columns_to_combine = ['title','company_profile', 'description', 'requirements', 'benefits']
data['text'] = data[columns_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
data = data.drop_duplicates(subset='text', keep='first')
codes, uniques = pd.factorize(data['description'])
X = data['text']
y = data['fraudulent']

missing_description = (codes == -1).sum()
empty_description = (data['description'].str.strip() == "").sum()
#group shuffle split
gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=42)

train_pos, temp_pos = next(gss.split(X, groups=codes))
X_train = X.iloc[train_pos]
X_temp = X.iloc[temp_pos]
y_train = y.iloc[train_pos]
y_temp = y.iloc[temp_pos]

temp_codes = codes[temp_pos]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_pos, test_pos = next(gss2.split(X_temp, groups=temp_codes))
X_val = X_temp.iloc[val_pos]
X_test = X_temp.iloc[test_pos]
y_val = y_temp.iloc[val_pos]
y_test = y_temp.iloc[test_pos]

vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

X_train_counts = vectorizer.fit_transform(X_train)
X_val_counts = vectorizer.transform(X_val)
X_train_counts_tf = tfidf_vectorizer.fit_transform(X_train)
X_val_counts_tf = tfidf_vectorizer.transform(X_val)
#initialize model
model = MultinomialNB()
model2 = MultinomialNB(alpha=0.001)
model.fit(X_train_counts, y_train)
labels = model.predict(X_val_counts)
model2.fit(X_train_counts_tf, y_train)
labels2 = model2.predict(X_val_counts_tf)

#confusion matrix
cm = confusion_matrix(y_val, labels)
cm2 = confusion_matrix(y_val, labels2)
print("\tCount Vectorizer\n")
print(classification_report(y_val, labels))
print("\tTFID Vectorizer\n")
print(classification_report(y_val, labels2))
#plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraudulent"])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Legitimate", "Fraudulent"])
disp.plot()
disp2.plot()
plt.show()