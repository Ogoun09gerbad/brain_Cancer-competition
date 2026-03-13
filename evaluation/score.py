import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def score_submission(y_true, y_pred):
    """Logic to calculate scores for the competition."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": accuracy,
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

if __name__ == '__main__':
    print("Scoring script initialized. Use score_submission(true, pred) to evaluate.")
