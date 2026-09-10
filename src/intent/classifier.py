import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from src.intent.data_loader import load_and_split_data
from src.intent.taxonomy import INTENT_CLASSES
from evaluation.intent_evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    train_ex, test_ex = load_and_split_data(random_state=42)
    
    X_train = [ex['text'] for ex in train_ex]
    y_train = [ex['weak_intent'] for ex in train_ex]
    
    X_test = [ex['text'] for ex in test_ex]
    y_test = [ex['weak_intent'] for ex in test_ex]
    
    logger.info("Training Improved Classifier (TF-IDF Bigrams + SGDClassifier)...")
    
    # We use character and word n-grams, and SGDClassifier for a lightweight but powerful baseline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000, tol=1e-3, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    logger.info("Predicting...")
    y_pred = pipeline.predict(X_test)
    
    evaluate_predictions(y_test, y_pred, INTENT_CLASSES, "classifier_improved", X_test)

if __name__ == "__main__":
    main()
