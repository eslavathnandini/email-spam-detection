import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
import joblib
import os


class ModelTrainer:
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.trained_models = {}
        self.results = {}
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models"""
        for model_name in self.trained_models.keys():
            self.evaluate_model(model_name, X_test, y_test)
        return self.results
    
    def get_best_model(self):
        """Get the best model based on F1 score"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        return best_model_name, self.trained_models[best_model_name]
    
    def cross_validate(self, model_name, X, y, cv=5):
        """Perform cross-validation"""
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        return scores.mean(), scores.std()
    
    def save_model(self, model_name, filepath):
        """Save a trained model"""
        model = self.trained_models[model_name]
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model = joblib.load(filepath)
        return model
    
    def get_results_dataframe(self):
        """Get results as a DataFrame"""
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('f1', ascending=False)
        return results_df


def train_spam_classifier(X_train, y_train, X_test, y_test):
    """Main function to train and evaluate spam classifier"""
    trainer = ModelTrainer()
    
    # Train all models
    trainer.train_all_models(X_train, y_train)
    
    # Evaluate all models
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\nBest Model: {best_name}")
    print(f"Results:")
    print(trainer.get_results_dataframe())
    
    return trainer, best_model


if __name__ == "__main__":
    # Test model training
    from sklearn.datasets import make_classification
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    trainer, best_model = train_spam_classifier(X_train, y_train, X_test, y_test)
