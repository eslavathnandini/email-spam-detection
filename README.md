# Email Spam Classifier

A machine learning-based email/SMS spam classifier that detects spam messages with **98.21% accuracy**.

## Project Overview

This project implements a spam detection system using Natural Language Processing (NLP) and Machine Learning techniques. The system classifies emails/SMS as spam or ham (not spam) based on text features.

## Problem Statement

**Goal:** Build an ML system to classify emails as spam or ham with high accuracy.

**Dataset:** SMS Spam Collection Dataset (5,572 messages)
- Source: UCI Machine Learning Repository / Kaggle
- Features: Label (spam/ham), Message text
- Distribution: 4,825 ham, 747 spam

## Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Deployment:** Streamlit

## Project Structure

```
email-spam-detection/
├── data/
│   └── spam.csv                    # Dataset (5,572 messages)
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Modeling.ipynb
├── src/
│   ├── text_preprocessing.py       # Text cleaning functions
│   ├── feature_engineering.py      # Feature extraction
│   └── model_training.py           # Model training
├── models/
│   ├── spam_classifier.pkl         # Trained SVM model
│   └── tfidf.pkl                   # TF-IDF vectorizer
├── run.py                          # Training script
├── app.py                          # Streamlit web app
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/eslavathnandini/email-spam-detection.git
cd email-spam-detection
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run training script
```bash
python run.py
```

### Step 4: Run Streamlit app
```bash
streamlit run app.py
```

## Results (Actual)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 96.23% | 100.0% | 71.81% | 83.59% |
| **SVM** | **98.21%** | **98.5%** | **87.92%** | **92.91%** |
| Random Forest | 96.95% | 100.0% | 77.18% | 87.12% |
| Logistic Regression | 97.4% | 100.0% | 80.54% | 89.22% |
| XGBoost | 97.4% | 96.88% | 83.22% | 89.53% |

**Best Model:** SVM with 98.21% accuracy and 92.91% F1-score

## Features

### Text Preprocessing
- Lowercase conversion
- Special character removal
- Stopword removal
- Text normalization

### Feature Engineering
- TF-IDF Vectorization (5,000 max features)
- Text statistics (length, word count, etc.)

### Models Implemented
- Multinomial Naive Bayes
- Support Vector Machine (SVM) - Best
- Random Forest
- Logistic Regression
- XGBoost

## Usage

### 1. Run Training Script
```bash
python run.py
```

This will:
- Load the dataset
- Preprocess text
- Train 5 ML models
- Display accuracy results
- Save the best model and TF-IDF vectorizer

### 2. Run Streamlit App
```bash
streamlit run app.py
```

This will:
- Open web browser
- Show spam classifier interface
- Allow real-time predictions

### 3. Use as Python Module
```python
from src.text_preprocessing import TextPreprocessor
import joblib

# Load model and TF-IDF
model = joblib.load('models/spam_classifier.pkl')
tfidf = joblib.load('models/tfidf.pkl')
preprocessor = TextPreprocessor()

# Predict
message = "CONGRATULATIONS! You've won $1000!"
cleaned = preprocessor.preprocess(message)
features = tfidf.transform([cleaned])
prediction = model.predict(features)[0]

print("SPAM" if prediction == 1 else "HAM")
```

## Key Learnings

1. **Text Preprocessing:** Importance of cleaning text data for ML
2. **Feature Extraction:** TF-IDF vs Bag of Words comparison
3. **Model Selection:** Comparing multiple classifiers
4. **Imbalanced Data:** Handling class imbalance in spam detection

## Resources

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://streamlit.io/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn for the ML libraries
- Streamlit for the web framework
