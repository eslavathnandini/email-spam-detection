"""
Email Spam Classifier - Run Script
Run this script to train models and get actual accuracy values
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import os
from src.text_preprocessing import TextPreprocessor


def main():
    print("="*70)
    print("EMAIL SPAM CLASSIFIER - TRAINING SCRIPT")
    print("="*70)
    
    # Check if dataset exists
    if not os.path.exists('data/spam.csv'):
        print("\nERROR: Dataset not found!")
        print("Please download spam.csv from:")
        print("https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
        print("And place it in the data/ folder")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('data/spam.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Preprocess text
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()
    df['cleaned'] = df['message'].apply(preprocessor.preprocess)
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['label_encoded'], test_size=0.2, random_state=42, stratify=df['label_encoded']
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # TF-IDF
    print("\nApplying TF-IDF vectorization...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train models
    print("\nTraining models...")
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = []
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    # Display results
    print("\n" + "="*70)
    print("MODEL RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    results_df['Accuracy'] = results_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    results_df['Precision'] = results_df['Precision'].apply(lambda x: f"{x*100:.2f}%")
    results_df['Recall'] = results_df['Recall'].apply(lambda x: f"{x*100:.2f}%")
    results_df['F1-Score'] = results_df['F1-Score'].apply(lambda x: f"{x*100:.2f}%")
    
    print(results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model_name}")
    print("="*70)
    
    # Save best model and TF-IDF
    print("\nSaving model and TF-IDF vectorizer...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(models[best_model_name], 'models/spam_classifier.pkl')
    joblib.dump(tfidf, 'models/tfidf.pkl')
    print("Model saved to models/spam_classifier.pkl")
    print("TF-IDF saved to models/tfidf.pkl")
    
    # Test with sample messages
    print("\n" + "="*70)
    print("TEST PREDICTIONS")
    print("="*70)
    
    sample_messages = [
        "Hey, how are you doing today?",
        "CONGRATULATIONS! You've won $1000! Click here to claim NOW!!!",
        "Meeting scheduled for tomorrow at 3pm",
        "FREE FREE FREE! Call now to claim your prize!",
        "Can you pick up groceries on your way home?"
    ]
    
    for msg in sample_messages:
        cleaned = preprocessor.preprocess(msg)
        vectorized = tfidf.transform([cleaned])
        prediction = models[best_model_name].predict(vectorized)[0]
        probability = models[best_model_name].predict_proba(vectorized)[0]
        
        print(f"\nMessage: {msg[:50]}...")
        print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
        print(f"Confidence: {max(probability)*100:.2f}%")
        print("-" * 50)
    
    print("\nTraining complete!")
    print("Run 'streamlit run app.py' to start the web app")


if __name__ == "__main__":
    main()
