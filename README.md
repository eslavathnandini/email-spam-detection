# Email Spam Classifier

A machine learning-based email/SMS spam classifier that detects spam messages with ~97% accuracy.

## 📋 Project Overview

This project implements a spam detection system using Natural Language Processing (NLP) and Machine Learning techniques. The system classifies emails/SMS as spam or ham (not spam) based on text features.

## 🎯 Problem Statement

**Goal:** Build an ML system to classify emails as spam or ham with high accuracy.

**Dataset:** SMS Spam Collection Dataset (5,572 messages)
- Source: UCI Machine Learning Repository / Kaggle
- Features: Label (spam/ham), Message text
- Distribution: 747 spam, 4825 ham

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, XGBoost
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Deployment:** Streamlit

## 📁 Project Structure

```
email-spam-classifier/
├── data/
│   ├── spam.csv                    # Raw dataset
│   └── preprocessed_spam.csv       # Preprocessed data
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Modeling.ipynb
├── src/
│   ├── text_preprocessing.py       # Text cleaning functions
│   ├── feature_engineering.py      # Feature extraction
│   └── model_training.py           # Model training
├── models/
│   └── spam_classifier.pkl         # Trained model
├── outputs/                        # Output files
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📊 Usage

### 1. Run Jupyter Notebooks

```bash
# EDA and Preprocessing
jupyter notebook notebooks/01_EDA_and_Preprocessing.ipynb

# Model Training
jupyter notebook notebooks/02_Modeling.ipynb
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

### 3. Use as Python Module

```python
from src.text_preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
import joblib

# Load model
model = joblib.load('models/spam_classifier.pkl')
preprocessor = TextPreprocessor()
feature_engineer = FeatureEngineer()

# Predict
message = "CONGRATULATIONS! You've won $1000!"
cleaned = preprocessor.preprocess(message)
features = feature_engineer.transform([cleaned])
prediction = model.predict(features)[0]

print("SPAM" if prediction == 1 else "HAM")
```

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.2% | 97.1% | 97.2% | 97.1% |
| SVM | 97.8% | 97.7% | 97.8% | 97.7% |
| XGBoost | 97.5% | 97.4% | 97.5% | 97.4% |

**Best Model:** SVM with 97.8% accuracy

## 🔧 Features

### Text Preprocessing
- Lowercase conversion
- Special character removal
- Stopword removal
- Stemming/Lemmatization

### Feature Engineering
- TF-IDF Vectorization
- Bag of Words
- Text statistics (length, word count, etc.)

### Models Implemented
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- XGBoost

## 📝 Key Learnings

1. **Text Preprocessing:** Importance of cleaning text data for ML
2. **Feature Extraction:** TF-IDF vs Bag of Words comparison
3. **Model Selection:** Comparing multiple classifiers
4. **Imbalanced Data:** Handling class imbalance in spam detection

## 🎓 Interview Questions

1. **How does TF-IDF work?**
   - TF-IDF weighs words by their frequency in a document vs. across all documents

2. **Why Naive Bayes for spam detection?**
   - Works well with text data, fast training, good baseline

3. **How to handle imbalanced data?**
   - SMOTE, undersampling, class weights

4. **What features did you engineer?**
   - TF-IDF, message length, word count, uppercase count

## 📚 Resources

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn for the ML libraries
- Streamlit for the web framework
