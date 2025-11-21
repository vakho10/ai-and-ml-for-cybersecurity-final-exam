"""
Spam Email Classifier
A machine learning model to distinguish between spam and non-spam (ham) emails.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import re
import string
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

class SpamClassifier:
    """
    A spam email classifier using machine learning.
    """

    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the classifier.

        Args:
            model_type: Type of model to use ('naive_bayes', 'logistic', 'svm')
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.model = self._get_model(model_type)
        self.is_trained = False

    def _get_model(self, model_type):
        """Get the specified model."""
        models = {
            'naive_bayes': MultinomialNB(),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='linear', random_state=42)
        }
        return models.get(model_type, MultinomialNB())

    def preprocess_text(self, text):
        """
        Preprocess email text.

        Args:
            text: Raw email text

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def train(self, X_train, y_train):
        """
        Train the classifier.

        Args:
            X_train: Training email texts
            y_train: Training labels (0=ham, 1=spam)
        """
        # Preprocess texts
        X_train_clean = [self.preprocess_text(text) for text in X_train]

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)

        # Train model
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        print(f"Model trained successfully using {self.model_type}")

    def predict(self, X_test):
        """
        Make predictions on new emails.

        Args:
            X_test: Email texts to classify

        Returns:
            Predictions (0=ham, 1=spam)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Preprocess and vectorize
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        X_test_vec = self.vectorizer.transform(X_test_clean)

        # Predict
        return self.model.predict(X_test_vec)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test email texts
            y_test: True labels

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        return metrics, y_pred

    def save_model(self, filepath):
        """Save the trained model."""
        with open(filepath, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.is_trained = True
        print(f"Model loaded from {filepath}")


def load_dataset():
    """
    Load or create a sample spam dataset.
    For demonstration purposes, we'll create a synthetic dataset.
    In practice, you would load from a CSV file.
    """
    # Sample spam emails
    spam_emails = [
        "Congratulations! You've won $1,000,000! Click here now!",
        "Get viagra at lowest prices. Order now!",
        "URGENT: Your account will be closed. Verify now!",
        "Make money fast! Work from home opportunity!",
        "Free loan approval in 24 hours. Apply today!",
        "Win a free iPhone! Limited time offer!",
        "Your package is waiting. Click to claim prize!",
        "Hot singles in your area want to meet you!",
        "Lose weight fast with this miracle pill!",
        "Earn $5000 per week working from home!"
    ] * 50  # Repeat to get more samples

    # Sample ham (non-spam) emails
    ham_emails = [
        "Hi, can we schedule a meeting for tomorrow at 3pm?",
        "Thanks for your help with the project yesterday.",
        "The quarterly report is attached. Please review.",
        "Your order has been shipped and will arrive Friday.",
        "Reminder: Team lunch at 12:30 in conference room B.",
        "Happy birthday! Hope you have a great day!",
        "The document you requested is now available.",
        "Please find attached the invoice for this month.",
        "Let me know if you need any clarification on the proposal.",
        "Looking forward to our call next week."
    ] * 50

    # Combine and create labels
    emails = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)

    # Create DataFrame
    df = pd.DataFrame({'text': emails, 'label': labels})

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def visualize_results(y_test, y_pred, metrics, output_dir='.'):
    """
    Create visualizations of model performance.
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    bars = plt.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}/ folder")


def main():
    """
    Main function to demonstrate the spam classifier.
    """
    print("=" * 60)
    print("Spam Email Classifier - Training and Evaluation")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading dataset...")
    df = load_dataset()
    print(f"   Total emails: {len(df)}")
    print(f"   Spam emails: {sum(df['label'] == 1)}")
    print(f"   Ham emails: {sum(df['label'] == 0)}")

    # Split data
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")

    # Train model
    print("\n3. Training the classifier...")
    classifier = SpamClassifier(model_type='naive_bayes')
    classifier.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model performance...")
    metrics, y_pred = classifier.evaluate(X_test, y_test)

    print("\n   Results:")
    print(f"   ├─ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   ├─ Precision: {metrics['precision']:.4f}")
    print(f"   ├─ Recall:    {metrics['recall']:.4f}")
    print(f"   └─ F1-Score:  {metrics['f1_score']:.4f}")

    # Classification report
    print("\n5. Detailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Ham', 'Spam']))

    # Create visualizations
    print("\n6. Creating visualizations...")
    visualize_results(y_test, y_pred, metrics)

    # Save model
    print("\n7. Saving model...")
    classifier.save_model('spam_classifier_model.pkl')

    # Test with custom examples
    print("\n8. Testing with custom examples:")
    test_emails = [
        "Congratulations! You won a lottery!",
        "Hey, can we meet tomorrow for coffee?"
    ]

    predictions = classifier.predict(test_emails)
    for email, pred in zip(test_emails, predictions):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"\n   Email: '{email}'")
        print(f"   Prediction: {label}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()