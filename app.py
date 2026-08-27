import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from src.text_preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer

# Load model and components
@st.cache_resource
def load_model():
    model = joblib.load('models/spam_classifier.pkl')
    preprocessor = TextPreprocessor()
    return model, preprocessor

# Load and fit feature engineer on training data
@st.cache_resource
def load_feature_engineer():
    # Load raw data and preprocess
    df = pd.read_csv('data/spam.csv')
    preprocessor = TextPreprocessor()
    df['cleaned'] = df['message'].apply(preprocessor.preprocess)
    
    feature_engineer = FeatureEngineer(method='tfidf', max_features=5000)
    feature_engineer.fit_transform(df['cleaned'])
    return feature_engineer

def main():
    st.set_page_config(
        page_title="Email Spam Classifier",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Email Spam Classifier")
    st.markdown("---")
    
    # Load components
    model, preprocessor = load_model()
    feature_engineer = load_feature_engineer()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app classifies emails/SMS as spam or ham (not spam) "
        "using machine learning.\n\n"
        "**Dataset:** SMS Spam Collection\n"
        "**Model:** SVM\n"
        "**Accuracy:** 98.21%"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Message")
        user_input = st.text_area(
            "Type your email or SMS message here:",
            height=150,
            placeholder="Enter message to classify..."
        )
        
        # Example messages
        st.markdown("**Example Messages:**")
        examples = [
            "Hey, how are you doing today?",
            "CONGRATULATIONS! You've won $1000! Click here to claim NOW!!!",
            "Meeting scheduled for tomorrow at 3pm",
            "FREE FREE FREE! Call now to claim your prize!"
        ]
        
        for example in examples:
            if st.button(f"Try: {example[:30]}...", key=example):
                user_input = example
                st.rerun()
    
    with col2:
        st.subheader("Prediction")
        
        if user_input:
            # Preprocess
            cleaned_text = preprocessor.preprocess(user_input)
            
            # Feature extraction
            features = feature_engineer.transform([cleaned_text])
            
            # Predict
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Display result
            if prediction == 1:
                st.error("🚨 **SPAM DETECTED**")
                st.metric("Confidence", f"{probability[1]*100:.2f}%")
            else:
                st.success("✅ **NOT SPAM (HAM)**")
                st.metric("Confidence", f"{probability[0]*100:.2f}%")
            
            # Show details
            st.markdown("---")
            st.markdown("**Analysis Details:**")
            st.write(f"- Message Length: {len(user_input)} characters")
            st.write(f"- Word Count: {len(user_input.split())} words")
            st.write(f"- Uppercase Count: {sum(1 for c in user_input if c.isupper())}")
            st.write(f"- Exclamation Marks: {user_input.count('!')}")
    
    # Batch prediction
    st.markdown("---")
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with 'message' column", type=['csv'])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        
        if 'message' in batch_df.columns:
            # Preprocess
            batch_df['cleaned'] = batch_df['message'].apply(preprocessor.preprocess)
            
            # Feature extraction
            features = feature_engineer.transform(batch_df['cleaned'])
            
            # Predict
            batch_df['prediction'] = model.predict(features)
            batch_df['confidence'] = model.predict_proba(features).max(axis=1)
            batch_df['label'] = batch_df['prediction'].map({0: 'HAM', 1: 'SPAM'})
            
            # Display results
            st.dataframe(batch_df[['message', 'label', 'confidence']])
            
            # Download results
            csv = batch_df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="spam_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("CSV file must contain a 'message' column")

if __name__ == "__main__":
    main()
