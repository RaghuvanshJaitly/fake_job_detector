# Fake Job Detector

Flags potentially fraudulent job postings for review using TF-IDF and Multinomial Naive Bayes.

## Run locally

Run these commands from the project root.

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Add the dataset

Place `fake_job_postings.csv` in the project root, alongside `train_model.py`.

### 3. Train the model

```bash
python3 train_model.py
```

Training saves the fitted pipeline and decision threshold to `fake_job_detector.joblib`.

### 4. Run the prediction demo

```bash
python3 predict.py
```

The demo loads the saved model and prints a fraud score and review flag for each sample posting.

### 5. Run the tests

```bash
python3 -m pytest
```

The prediction test checks that the function returns one result per posting in order and flags scores at or above the threshold.