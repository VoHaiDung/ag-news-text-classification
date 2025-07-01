import os
import numpy as np
import joblib
import argparse
import logging
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return logging.getLogger(__name__)


logger = configure_logger()

# Load multiple logits files and corresponding labels
def load_logits_and_labels(logits_paths: List[str], labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    logits_list = [np.load(path) for path in logits_paths]
    X = np.concatenate(logits_list, axis=1)
    y = np.load(labels_path)
    return X, y

# Build a stacking classifier with base and meta learners
def build_stacking_classifier(cv: int = 5) -> StackingClassifier:
    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ("svm", SVC(kernel="linear", probability=True, C=1.0, random_state=42)),
    ]
    meta_learner = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=-1
    )
    return clf

# Evaluate the trained model on input data
def evaluate_model(model, X: np.ndarray, y: np.ndarray, class_names: List[str]) -> None:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)

    logger.info("Evaluation Metrics")
    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  Precision: %.4f", precision)
    logger.info("  Recall:    %.4f", recall)
    logger.info("  F1-score:  %.4f", f1)

    report = classification_report(y, preds, target_names=class_names, zero_division=0)
    logger.info("Classification Report:\n%s", report)


def main():
    parser = argparse.ArgumentParser(description="Train stacking ensemble using base model logits")
    parser.add_argument("--logits", nargs="+", required=True,
                        help="Paths to .npy files of model logits (e.g. logits_deberta.npy)")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to .npy file containing true labels")
    parser.add_argument("--save_model", type=str, default="outputs/checkpoints/stacking_model.joblib",
                        help="Path to save trained stacking model")
    parser.add_argument("--class_names", nargs="+", default=["World", "Sports", "Business", "Sci/Tech"])
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    logger.info("Loading logits and labels...")
    X, y = load_logits_and_labels(args.logits, args.labels)
    logger.info("Logits shape: %s | Labels shape: %s", X.shape, y.shape)

    clf = build_stacking_classifier(cv=args.cv)
    logger.info("Training stacking classifier...")
    clf.fit(X, y)

    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    joblib.dump(clf, args.save_model)
    logger.info("Saved stacking model to %s", args.save_model)

    logger.info("Evaluating model on training set:")
    evaluate_model(clf, X, y, class_names=args.class_names)


if __name__ == "__main__":
    main()
