import os
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Dataset added here
DATA_PATH = 'mail_data.csv'

def _build_pipeline():
    tfidf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, ngram_range=(1, 2), max_features=10000)
    pipe_lr = Pipeline([('tfidf', tfidf), ('lr', LogisticRegression(max_iter=1000, solver='liblinear'))])
    return pipe_lr

def _train_pipeline(pipe: Pipeline):
    
    if os.path.exists(DATA_PATH):
        logger.info('Loading dataset from %s', DATA_PATH)
        raw = pd.read_csv(DATA_PATH)
        data = raw.where(pd.notnull(raw), '')
        if 'Category' in data.columns and 'Message' in data.columns:
            data['Category'] = data['Category'].astype(str).str.strip().str.lower()
            data.loc[data['Category'] == 'spam', 'label'] = 1
            data.loc[data['Category'] == 'ham', 'label'] = 0
            data['label'] = data['label'].astype(int)
            X = data['Message']
            y = data['label']
        else:
            logger.warning('Dataset missing expected columns. Using fallback data.')
            X, y = _fallback_data()
    else:
        logger.info('No dataset found at %s. Using fallback data.', DATA_PATH)
        X, y = _fallback_data()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(x_train, y_train)
    logger.info('Training complete.')
    return pipe


def _fallback_data():
    # Small example dataset so the app can run without external files.
    messages = [
        'Win a free iPhone now',
        'Limited time offer, claim your prize',
        'You have been selected for a $1000 gift card',
        'Reminder: project meeting tomorrow at 10 AM',
        'Can we reschedule our call?',
        'Lunch at noon?'
    ]
    labels = [1, 1, 1, 0, 0, 0]
    return pd.Series(messages), pd.Series(labels)


pipe = _build_pipeline()
pipe = _train_pipeline(pipe)


def predict(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        text = str(text)

    proba = None
    label = 'Unknown'
    try:
        probs = pipe.predict_proba([text])[0]
        
        if len(probs) == 2:
            spam_prob = float(probs[1])
        else:
        
            spam_prob = float(probs[0])
        is_spam = spam_prob >= 0.5
        label = 'Spam' if is_spam else 'Ham'
        confidence = max(spam_prob, 1.0 - spam_prob)
        proba = spam_prob
    except Exception:
        pred = pipe.predict([text])[0]
        is_spam = int(pred) == 1
        label = 'Spam' if is_spam else 'Ham'
        confidence = 1.0
        proba = float(is_spam)

    return {
        'label': label,
        'is_spam': bool(is_spam),
        'confidence': float(confidence),
        'spam_probability': float(proba),
    }


if __name__ == '__main__':
    # test
    sample = 'Hi John, just a reminder that our project meeting is scheduled for tomorrow at 10 AM.'
    print(predict(sample))
