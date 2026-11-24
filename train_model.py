import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, average_precision_score, PrecisionRecallDisplay
from pipeline import build_pipeline
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_at_k

def prepare_data(data: pd.DataFrame):
    df = data.copy()
    columns_to_combine = ['title','company_profile', 'description', 'requirements', 'benefits']
    df['text'] = df[columns_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
    df = df.drop_duplicates(subset='text', keep='first')

    return df

def split_data(data: pd.DataFrame):
    df = data.copy()
    codes, uniques = pd.factorize(df['description'])
    X = df['text']
    y = df['fraudulent']
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    pd.set_option('display.max_colwidth', None)
    threshold = 0.10
    data = pd.read_csv('fake_job_postings.csv', sep=',')
    prepared_data = prepare_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(prepared_data)

    multinomial_model = MultinomialNB(alpha=0.001)
    logistic_model = LogisticRegression()

    model_pipeline_mb = build_pipeline(multinomial_model)
    model_pipeline_logistic = build_pipeline(logistic_model)
    model_pipeline_mb.fit(X_train, y_train)
    model_pipeline_logistic.fit(X_train, y_train)
    prob_mb = model_pipeline_mb.predict_proba(X_val)
    prob_logistic = model_pipeline_logistic.predict_proba(X_val)

    model_artifact = {"pipeline": model_pipeline_mb, "threshold": threshold}
    joblib.dump(model_artifact, 'fake_job_detector.joblib')

    pos_prob_mb = prob_mb[:, 1]
    pos_prob_logistic = prob_logistic[:, 1]
    mb_predictions = pos_prob_mb >= threshold
    logistic_predictions = pos_prob_logistic >= threshold

    # Compare how many actual frauds each model catches when we
    # review only its top k highest-scoring postings.
    k = 119
    result_mb = evaluate_at_k(y_val, pos_prob_mb, k)
    result_logistic = evaluate_at_k(y_val, pos_prob_logistic, k)
    
    print(f"Multinomial Naive Bayes (evaluate k postings): {result_mb}")
    print(f"Logistic Regression (evaluate k postings): {result_logistic}")
    
    #average precision scores
    avg_precision_nb = average_precision_score(y_val, pos_prob_mb)
    avg_precision_logistic = average_precision_score(y_val, pos_prob_logistic)

    #confusion matrix
    cm_mb = confusion_matrix(y_val, mb_predictions)
    cm_logistic = confusion_matrix(y_val, logistic_predictions)

    print(f"\tTFID Vectorizer Multinomial NB with {threshold} threshold\n")
    print(classification_report(y_val, mb_predictions))
    print(f"\tTFID Vectorizer Logistic regression with {threshold} threshold\n")
    print(classification_report(y_val, logistic_predictions))

    print(f"Average Precision MultinomialNB: {avg_precision_nb}\n")
    print(f"Average Precision Logistic Regression: {avg_precision_logistic}\n")
    #plot the matrix
    display_mb = ConfusionMatrixDisplay(confusion_matrix=cm_mb, display_labels=["Legitimate", "Fraudulent"])
    display_logistic = ConfusionMatrixDisplay(confusion_matrix=cm_logistic, display_labels=["Legitimate", "Fraudulent"])

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    display_mb.plot(ax=axes[0])
    axes[0].set_title("Multinomial Naive Bayes")

    display_logistic.plot(ax=axes[1])
    axes[1].set_title("Logistic Regression")
    plt.tight_layout()
    
    pr_fig, pr_ax = plt.subplots(figsize=(7,5))
    PrecisionRecallDisplay.from_predictions(
        y_val,
        pos_prob_mb,
        name="Multinomial NB",
        ax=pr_ax,
    )
    PrecisionRecallDisplay.from_predictions(
        y_val,
        pos_prob_logistic,
        name="Logistic Regression",
        ax=pr_ax,
    )
    pr_ax.set_title("Validation precision-recall curves")
    pr_fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    
    main()