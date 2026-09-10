import logging
from collections import Counter
from src.intent.data_loader import load_and_split_data
from src.intent.taxonomy import INTENT_CLASSES
from evaluation.intent_evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    train_ex, test_ex = load_and_split_data()
    
    # Find majority class in train
    train_labels = [ex['weak_intent'] for ex in train_ex]
    majority_class = Counter(train_labels).most_common(1)[0][0]
    logger.info(f"Majority class: {majority_class}")
    
    y_true = [ex['weak_intent'] for ex in test_ex]
    y_pred = [majority_class] * len(test_ex)
    texts = [ex['text'] for ex in test_ex]
    
    evaluate_predictions(y_true, y_pred, INTENT_CLASSES, "baseline_majority", texts)

if __name__ == "__main__":
    main()
