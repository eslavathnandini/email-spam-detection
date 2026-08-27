import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import re


class FeatureEngineer:
    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features)
    
    def extract_basic_features(self, df):
        """Extract basic text features"""
        df = df.copy()
        
        # Message length
        df['message_length'] = df['message'].apply(len)
        
        # Word count
        df['word_count'] = df['message'].apply(lambda x: len(x.split()))
        
        # Uppercase count
        df['uppercase_count'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()))
        
        # Exclamation count
        df['exclamation_count'] = df['message'].apply(lambda x: x.count('!'))
        
        # Question mark count
        df['question_count'] = df['message'].apply(lambda x: x.count('?'))
        
        # Digit count
        df['digit_count'] = df['message'].apply(lambda x: sum(1 for c in x if c.isdigit()))
        
        # Special character count
        df['special_char_count'] = df['message'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()))
        
        return df
    
    def fit_transform(self, texts):
        """Fit and transform texts to vectors"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts to vectors"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        return self.vectorizer.get_feature_names_out()
    
    def create_dataframe(self, vectors, prefix='tfidf'):
        """Create dataframe from sparse matrix"""
        feature_names = self.get_feature_names()
        df = pd.DataFrame(vectors.toarray(), columns=[f'{prefix}_{name}' for name in feature_names])
        return df


class TextAnalyzer:
    """Analyze text patterns"""
    
    @staticmethod
    def get_word_frequency(texts, top_n=20):
        """Get word frequency from texts"""
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        return Counter(all_words).most_common(top_n)
    
    @staticmethod
    def get_avg_word_length(text):
        """Get average word length"""
        words = text.split()
        if not words:
            return 0
        return np.mean([len(word) for word in words])
    
    @staticmethod
    def get_unique_word_ratio(text):
        """Get ratio of unique words to total words"""
        words = text.split()
        if not words:
            return 0
        return len(set(words)) / len(words)


if __name__ == "__main__":
    # Test feature engineering
    sample_texts = [
        "Hey! How are you doing today?",
        "CONGRATULATIONS! You've won $1000! Click here to claim NOW!!!",
        "Meeting scheduled for tomorrow at 3pm"
    ]
    
    # Test basic features
    df = pd.DataFrame({'message': sample_texts})
    engineer = FeatureEngineer()
    df_with_features = engineer.extract_basic_features(df)
    print("Basic Features:")
    print(df_with_features[['message', 'message_length', 'word_count', 'uppercase_count']])
    
    # Test TF-IDF
    print("\nTF-IDF Features:")
    tfidf_matrix = engineer.fit_transform(sample_texts)
    print(f"Shape: {tfidf_matrix.shape}")
