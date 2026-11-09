import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from pipeline import build_pipeline
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

threshold = 0.10
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

multinomial_model = MultinomialNB(alpha=0.001)
logistic_model = LogisticRegression()

model_pipeline_mb = build_pipeline(multinomial_model)
model_pipeline_logistic = build_pipeline(logistic_model)
model_pipeline_mb.fit(X_train, y_train)
model_pipeline_logistic.fit(X_train, y_train)
labels_mb = model_pipeline_mb.predict(X_val)
labels_logistic = model_pipeline_logistic.predict(X_val)
prob_mb = model_pipeline_mb.predict_proba(X_val)
prob_logistic = model_pipeline_logistic.predict_proba(X_val)

model_artifact = {"pipeline": model_pipeline_mb, "threshold": threshold}
joblib.dump(model_artifact, 'fake_job_detector.joblib')

pos_prob_mb = prob_mb[:, 1]
pos_prob_logistic = prob_logistic[:, 1]
mb_predictions = pos_prob_mb >= threshold
false_neg = (y_val == 1) & (mb_predictions== 0)
logistic_predictions = pos_prob_logistic >= threshold

#confusion matrix
cm_mb = confusion_matrix(y_val, mb_predictions)
cm_logistic = confusion_matrix(y_val, logistic_predictions)

print(f"\tTFID Vectorizer Multinomial NB with {threshold} threshold\n")
print(classification_report(y_val, mb_predictions))
print(f"\tTFID Vectorizer Logistic regression with {threshold} threshold\n")
print(classification_report(y_val, logistic_predictions))

#plot the matrix
display_mb = ConfusionMatrixDisplay(confusion_matrix=cm_mb, display_labels=["Legitimate", "Fraudulent"])
display_logistic = ConfusionMatrixDisplay(confusion_matrix=cm_logistic, display_labels=["Legitimate", "Fraudulent"])
display_mb.plot()
display_logistic.plot()
plt.show()