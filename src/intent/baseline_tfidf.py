import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    
    logger.info("Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info("Training Logistic Regression baseline...")
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)
    
    logger.info("Predicting...")
    y_pred = clf.predict(X_test_vec)
    
    evaluate_predictions(y_test, y_pred, INTENT_CLASSES, "baseline_tfidf", X_test)

if __name__ == "__main__":
    main()
