import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from pipeline import build_pipeline
import joblib

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

model_pipeline = build_pipeline()
model_pipeline.fit(X_train, y_train)
labels = model_pipeline.predict(X_val)
prob = model_pipeline.predict_proba(X_val)
model_artifact = {"pipeline": model_pipeline, "threshold": threshold}
joblib.dump(model_artifact, 'fake_job_detector.joblib')

pos_prob = prob[:, 1]
new_predictions = pos_prob >= threshold
false_neg = (y_val == 1) & (new_predictions== 0)
print(pos_prob[false_neg])

#confusion matrix

cm2 = confusion_matrix(y_val, labels)
cm3 = confusion_matrix(y_val, new_predictions)
print("\tTFID Vectorizer\n")
print(classification_report(y_val, labels))
print(f"\tTFID Vectorizer with {threshold} threshold\n")
print(classification_report(y_val, new_predictions))
#plot the matrix

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Legitimate", "Fraudulent"])
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=["Legitimate", "Fraudulent"])
disp2.plot()
disp3.plot()
plt.show()