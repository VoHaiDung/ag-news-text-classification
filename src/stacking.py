import os
import argparse
import joblib
import numpy as np
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.utils import (
    configure_logger,
    load_logits_and_labels,
    print_classification_report,
)

# Initialize logger
logger = configure_logger("outputs/logs/stacking.log")

# Build stacking classifier with base and meta learners
# cv: number of folds for internal cross-validation
def build_stacking_classifier(cv: int = 5, passthrough: bool = False) -> StackingClassifier:
    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ("svm", SVC(kernel="linear", probability=True, random_state=42)),
    ]
    meta_learner = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=-1,
        passthrough=passthrough,
        verbose=1,
    )
    return clf

def main():
    parser = argparse.ArgumentParser(description="Train & evaluate stacking ensemble")
    parser.add_argument("--train_logits", nargs="+", required=True,
                        help="Paths to train logits .npy files")
    parser.add_argument("--train_labels", type=str, required=True,
                        help="Path to train labels .npy file")
    parser.add_argument("--test_logits", nargs="+",
                        help="Paths to test logits .npy files (for final evaluation)")
    parser.add_argument("--test_labels", type=str,
                        help="Path to test labels .npy file")
    parser.add_argument("--save_model", type=str,
                        default="outputs/checkpoints/stacking_model.joblib",
                        help="Path to save trained stacking model")
    parser.add_argument("--class_names", nargs="+",
                        default=["World", "Sports", "Business", "Sci/Tech"],
                        help="List of class names for reporting")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of CV folds for internal evaluation")
    parser.add_argument("--passthrough", action="store_true",
                        help="Include original features in meta learner")
    args = parser.parse_args()

    # Load train data
    logger.info("Loading train logits and labels...")
    X_train, y_train = load_logits_and_labels(args.train_logits, args.train_labels)
    logger.info(f"Train data shape: X={X_train.shape}, y={y_train.shape}")

    # Internal cross-validation
    clf = build_stacking_classifier(cv=args.cv, passthrough=args.passthrough)
    logger.info("Running internal %d-fold CV...", args.cv)
    scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42),
                              scoring="accuracy", n_jobs=-1)
    logger.info("CV Accuracy: mean=%.4f, std=%.4f", scores.mean(), scores.std())

    # Train on full train set
    logger.info("Training stacking classifier on full train set...")
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    joblib.dump(clf, args.save_model)
    logger.info(f"Saved stacking model to {args.save_model}")

    # Final evaluation on test set if provided
    if args.test_logits and args.test_labels:
        logger.info("Evaluating on test set...")
        X_test, y_test = load_logits_and_labels(args.test_logits, args.test_labels)
        logits_test = clf.predict_proba(X_test)
        print_classification_report(logits_test, y_test, tuple(args.class_names))
    else:
        logger.info("No test data provided, skipping final evaluation.")

if __name__ == "__main__":
    main()
