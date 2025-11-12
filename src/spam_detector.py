"""
Complete Spam Detection System
DLBAIPNLP01 – Project: NLP - Task 2

Implements text classification for spam detection using:
- Naive Bayes (baseline)
- Support Vector Machine (SVM)
- Deep Learning (optional)

Evaluates on progressively larger datasets as required by guidelines.

Author: Martin Lana Bengut
Date: November 2025
"""

import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
import joblib

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class SpamDetector:
    """Complete spam detection system"""
    
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, filepath='data/SMSSpamCollection'):
        """Load SMS Spam Collection dataset"""
        print("Loading dataset...")
        
        # Read tab-separated file
        df = pd.read_csv(filepath, sep='\t', names=['label', 'message'])
        
        # Convert labels to binary
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        print(f"✓ Loaded {len(df)} messages")
        print(f"  Spam: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
        print(f"  Ham:  {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
        
        return df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\d{10,}', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        
        # Stemming
        tokens = [self.stemmer.stem(w) for w in tokens]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df, sample_size=None):
        """Preprocess all messages"""
        print(f"\nPreprocessing messages...")
        
        # Sample if needed
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"  Using {sample_size} samples")
        
        # Preprocess
        df['processed'] = df['message'].apply(self.preprocess_text)
        
        # Remove empty messages
        df = df[df['processed'].str.len() > 0]
        
        print(f"✓ Preprocessed {len(df)} messages")
        
        return df
    
    def vectorize(self, X_train, X_test, method='tfidf'):
        """Convert text to numerical features"""
        print(f"\nVectorizing with {method.upper()}...")
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 2))
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Train shape: {X_train_vec.shape}")
        print(f"  Test shape: {X_test_vec.shape}")
        
        return X_train_vec, X_test_vec
    
    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes classifier"""
        print("\nTraining Naive Bayes...")
        
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        
        self.models['naive_bayes'] = model
        print("✓ Naive Bayes trained")
        
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM classifier"""
        print("\nTraining SVM...")
        
        model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        model.fit(X_train, y_train)
        
        self.models['svm'] = model
        print("✓ SVM trained")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics, y_pred, y_prob
    
    def predict(self, messages):
        """Predict spam for new messages"""
        if self.vectorizer is None or 'naive_bayes' not in self.models:
            raise ValueError("Model not trained yet")
        
        # Preprocess
        processed = [self.preprocess_text(msg) for msg in messages]
        
        # Vectorize
        X = self.vectorizer.transform(processed)
        
        # Predict
        predictions = self.models['naive_bayes'].predict(X)
        probabilities = self.models['naive_bayes'].predict_proba(X)
        
        return predictions, probabilities
    
    def save_model(self, model_name='naive_bayes', filepath='models/spam_detector.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.models.get(model_name),
            'vectorizer': self.vectorizer
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/spam_detector.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.models['naive_bayes'] = model_data['model']
        self.vectorizer = model_data['vectorizer']
        print(f"✓ Model loaded from {filepath}")

def main():
    """Main execution - Quick test"""
    detector = SpamDetector()
    
    # Load data
    df = detector.load_data()
    
    # Prepare small sample
    df_processed = detector.prepare_data(df, sample_size=1000)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df_processed['processed'], df_processed['label'],
        test_size=0.2, random_state=42, stratify=df_processed['label']
    )
    
    # Vectorize
    X_train_vec, X_test_vec = detector.vectorize(X_train, X_test)
    
    # Train
    detector.train_naive_bayes(X_train_vec, y_train)
    
    # Evaluate
    metrics, _, _ = detector.evaluate_model(
        detector.models['naive_bayes'], X_test_vec, y_test, 'Naive Bayes'
    )
    
    print("\n✓ Quick test complete")

if __name__ == "__main__":
    main()

