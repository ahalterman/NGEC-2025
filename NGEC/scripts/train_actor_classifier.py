"""
Train a multiclass classifier for actor code resolution.

This script:
1. Loads patterns from PLOVER_agents.txt
2. Generates embeddings using the sentence transformer model
3. Trains a logistic regression classifier
4. Evaluates performance on a test set
5. Saves the trained model

Usage:
    python scripts/train_actor_classifier.py
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_parse_agents(agents_file):
    """
    Load and parse the PLOVER agents file.

    This uses the same parsing logic as AgentMatcher._load_and_clean_agents()
    to ensure consistency.

    Args:
        agents_file: Path to PLOVER_agents.txt

    Returns:
        list: List of dicts with keys 'pattern', 'code_1', 'code_2'
    """
    logger.info(f"Loading agents from {agents_file}")

    with open(agents_file, "r", encoding="utf-8") as f:
        data = f.read()

    # Remove curly braces content
    data = re.sub(r"\{.+?\}", "", data)

    # Split into lines and filter
    lines = [line for line in data.split("\n") if line and not line.startswith("#")]
    lines = [re.sub(r"#.+", "", line).strip() for line in lines if not line.startswith("!")]

    logger.info(f"Found {len(lines)} raw lines")

    # Parse patterns
    patterns = []
    for line in lines:
        try:
            # Extract code from square brackets
            code_match = re.findall(r"\[.+?\]", line)
            if not code_match:
                continue

            code = re.sub(r"[\[\]~]", "", code_match[0]).strip()

            # Extract pattern
            pattern = re.sub(r"(\[.+?\])", "", line)
            pattern = re.sub(r"_", " ", pattern).lower().strip()

            patterns.append({
                "pattern": pattern,
                "code_1": code[0:3],
                "code_2": code[3:]
            })
        except Exception as e:
            logger.debug(f"Error loading {line}: {e}")

    # Handle special pattern replacements
    cleaned_patterns = []
    for pattern in patterns:
        if 'code_1' not in pattern:
            continue

        # Handle !minist! placeholder
        if re.search("!minist!", pattern['pattern']):
            for replacement in ["Minister", "Ministers", "Ministry", "Ministries"]:
                new_pattern = {
                    "code_1": pattern['code_1'],
                    "code_2": pattern['code_2'],
                    "pattern": re.sub(r"!minist!", replacement, pattern['pattern']).title()
                }
                cleaned_patterns.append(new_pattern)

        # Handle !person! placeholder
        elif re.search("!person!", pattern['pattern']):
            for replacement in ["person", "man", "woman", "men", "women"]:
                new_pattern = {
                    "code_1": pattern['code_1'],
                    "code_2": pattern['code_2'],
                    "pattern": re.sub(r"!person!", replacement, pattern['pattern'])
                }
                cleaned_patterns.append(new_pattern)
        else:
            cleaned_patterns.append(pattern)

    logger.info(f"Parsed {len(cleaned_patterns)} patterns after expansion")
    return cleaned_patterns


def generate_embeddings(patterns, model_name="jinaai/jina-embeddings-v3"):
    """
    Generate embeddings for all patterns.

    Args:
        patterns: List of pattern dicts
        model_name: Name of the sentence transformer model

    Returns:
        numpy.ndarray: Matrix of embeddings
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    logger.info(f"Generating embeddings for {len(patterns)} patterns...")
    pattern_texts = [p['pattern'] for p in patterns]
    embeddings = model.encode(pattern_texts, show_progress_bar=True, batch_size=32,
                              task="classification")

    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings, model


def train_classifier(X_train, y_train, X_test, y_test):
    """
    Train a logistic regression classifier.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels

    Returns:
        Trained classifier
    """
    logger.info("Training logistic regression classifier...")

    # Train with balanced class weights to handle imbalanced data
    clf = SVC(
        class_weight='balanced',
        probability=True,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    logger.info(f"Training accuracy: {train_acc:.3f}")
    logger.info(f"Test accuracy: {test_acc:.3f}")

    # Cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Detailed evaluation
    y_pred = clf.predict(X_test)
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf


def analyze_errors(clf, X_test, y_test, patterns_test, seed=0):
    """
    Analyze classification errors.

    Args:
        clf: Trained classifier
        X_test: Test embeddings
        y_test: True labels
        patterns_test: Test patterns (for inspection)
    """
    logger.info("\n" + "="*50)
    logger.info("ERROR ANALYSIS")
    logger.info("="*50)

    y_pred = clf.predict(X_test)
    errors = []

    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            proba = clf.predict_proba([X_test[i]])[0]
            true_prob = proba[list(clf.classes_).index(true)]
            pred_prob = proba[list(clf.classes_).index(pred)]

            errors.append({
                'pattern': patterns_test[i]['pattern'],
                'true': true,
                'pred': pred,
                'true_prob': true_prob,
                'pred_prob': pred_prob
            })

    if errors:
        logger.info(f"\nFound {len(errors)} errors out of {len(y_test)} test examples")
        logger.info("\nSample errors (first 20):")
        random.seed(seed)
        random.shuffle(errors)
        for err in errors[:20]:
            logger.info(f"  '{err['pattern']}' -> TRUE: {err['true']} (p={err['true_prob']:.2f}), PRED: {err['pred']} (p={err['pred_prob']:.2f})")
    else:
        logger.info("\nNo errors! Perfect classification on test set.")


def main():
    """Main training pipeline."""
    # Paths
    base_path = "NGEC/assets"
    agents_file = os.path.join(base_path, "PLOVER_agents.txt")
    output_model = os.path.join(base_path, "actor_classifier.pkl")
    output_classes = os.path.join(base_path, "actor_classifier_classes.txt")

    # Check if agents file exists
    if not os.path.exists(agents_file):
        logger.error(f"Agents file not found: {agents_file}")
        sys.exit(1)

    # Load patterns
    patterns = load_and_parse_agents(agents_file)

    # Show class distribution
    code_counts = Counter([p['code_1'] for p in patterns])
    logger.info("\nClass distribution:")
    for code, count in code_counts.most_common(100):
        logger.info(f"  {code}: {count}")

    # Remove classes with fewer than 5 examples
    valid_codes = {code for code, count in code_counts.items() if count >= 5}
    patterns = [p for p in patterns if p['code_1'] in valid_codes]
    logger.info(f"\nFiltered patterns to {len(patterns)} examples across {len(valid_codes)} classes")

    # Generate embeddings
    embeddings, embedding_model = generate_embeddings(patterns)

    # Prepare data
    X = embeddings
    y = np.array([p['code_1'] for p in patterns])

    # Split data (stratified to maintain class balance)
    logger.info("\nSplitting data into train/test (80/20)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(patterns)),
        test_size=0.4,
        stratify=y,
        random_state=42
    )

    patterns_test = [patterns[i] for i in idx_test]

    logger.info(f"Training set: {len(X_train)} examples")
    logger.info(f"Test set: {len(X_test)} examples")

    # Train classifier
    clf = train_classifier(X_train, y_train, X_test, y_test)

    # Analyze errors
    analyze_errors(clf, X_test, y_test, patterns_test, seed=42)

    # Now fit on full data
    logger.info("\nRefitting classifier on full dataset...")
    clf.fit(X, y)   

    # Save model
    logger.info(f"\nSaving classifier to {output_model}")
    with open(output_model, 'wb') as f:
        pickle.dump(clf, f)

    # Save class list for reference
    with open(output_classes, 'w') as f:
        for cls in clf.classes_:
            f.write(f"{cls}\n")

    logger.info(f"Saved class list to {output_classes}")
    logger.info("\nTraining complete!")
    logger.info(f"Model saved to: {output_model}")

    # Test a few examples
    logger.info("\n" + "="*50)
    logger.info("TESTING EXAMPLES")
    logger.info("="*50)

    test_examples = [
        "armed tribesmen",
        "soldier",
        "rebel fighter",
        "civilian",
        "police officer",
        "government official",
        "journalist"
    ]

    for example in test_examples:
        emb = embedding_model.encode([example])[0]
        pred = clf.predict([emb])[0]
        proba = clf.predict_proba([emb])[0]
        confidence = proba[list(clf.classes_).index(pred)]
        logger.info(f"  '{example}' -> {pred} (confidence: {confidence:.3f})")


if __name__ == '__main__':
    main()
