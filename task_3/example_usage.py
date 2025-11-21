"""
Example Usage Script for Spam Classifier
Demonstrates practical applications and use cases
"""

from spam_classifier import SpamClassifier
import pandas as pd


def example_1_quick_start():
    """
    Example 1: Quick start - Train and test with sample data
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Quick Start")
    print("=" * 60)

    from spam_classifier import load_dataset
    from sklearn.model_selection import train_test_split

    # Load data
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Train
    classifier = SpamClassifier(model_type='naive_bayes')
    classifier.train(X_train, y_train)

    # Test
    metrics, _ = classifier.evaluate(X_test, y_test)
    print(f"\nAccuracy: {metrics['accuracy']:.2%}")
    print(f"F1-Score: {metrics['f1_score']:.2%}")


def example_2_custom_emails():
    """
    Example 2: Classify your own emails
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Classify Custom Emails")
    print("=" * 60)

    from spam_classifier import load_dataset
    from sklearn.model_selection import train_test_split

    # Train model
    df = load_dataset()
    X_train, _, y_train, _ = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    classifier = SpamClassifier()
    classifier.train(X_train, y_train)

    # Test with custom emails
    test_emails = [
        "Hi John, can we reschedule our meeting to next Tuesday?",
        "CONGRATULATIONS!!! You won $10,000! Click here NOW!!!",
        "The project deadline has been extended to next Friday.",
        "Buy cheap meds online! No prescription needed!",
        "Please find attached the quarterly sales report.",
        "Lose 30 pounds in 30 days! Miracle weight loss pill!",
        "Your Amazon order #12345 has been shipped.",
        "URGENT: Your bank account will be closed! Verify now!",
        "Thanks for your help with the presentation yesterday.",
        "Get rich quick! Make $5000 per day from home!"
    ]

    predictions = classifier.predict(test_emails)

    print("\nEmail Classification Results:")
    print("-" * 60)
    for i, (email, pred) in enumerate(zip(test_emails, predictions), 1):
        label = "🚫 SPAM" if pred == 1 else "✅ HAM"
        # Truncate long emails
        display_email = email if len(email) <= 50 else email[:47] + "..."
        print(f"{i:2d}. [{label}] {display_email}")


def example_3_compare_algorithms():
    """
    Example 3: Compare different ML algorithms
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Compare Different Algorithms")
    print("=" * 60)

    from spam_classifier import load_dataset
    from sklearn.model_selection import train_test_split

    # Prepare data
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Test different algorithms
    algorithms = ['naive_bayes', 'logistic', 'svm']
    results = {}

    for algo in algorithms:
        print(f"\nTraining {algo.replace('_', ' ').title()}...")
        classifier = SpamClassifier(model_type=algo)
        classifier.train(X_train, y_train)
        metrics, _ = classifier.evaluate(X_test, y_test)
        results[algo] = metrics

    # Display comparison
    print("\n" + "-" * 60)
    print("Algorithm Performance Comparison:")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 60)

    for algo, metrics in results.items():
        print(f"{algo.replace('_', ' ').title():<20} "
              f"{metrics['accuracy']:<12.2%} "
              f"{metrics['f1_score']:<12.2%}")

    # Find best algorithm
    best_algo = max(results.items(), key=lambda x: x[1]['accuracy'])
    print("-" * 60)
    print(f"Best Algorithm: {best_algo[0].replace('_', ' ').title()} "
          f"(Accuracy: {best_algo[1]['accuracy']:.2%})")


def example_4_save_load_model():
    """
    Example 4: Save and load a trained model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Save and Load Model")
    print("=" * 60)

    from spam_classifier import load_dataset
    from sklearn.model_selection import train_test_split

    # Train and save model
    print("\nStep 1: Training model...")
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    classifier = SpamClassifier()
    classifier.train(X_train, y_train)

    print("Step 2: Saving model...")
    classifier.save_model('my_model.pkl')

    # Load and use model
    print("Step 3: Loading model...")
    new_classifier = SpamClassifier()
    new_classifier.load_model('my_model.pkl')

    print("Step 4: Testing loaded model...")
    test_email = ["Win a free iPhone! Click now!!!"]
    prediction = new_classifier.predict(test_email)
    result = "SPAM" if prediction[0] == 1 else "HAM"

    print(f"\nTest Email: '{test_email[0]}'")
    print(f"Prediction: {result}")
    print("\n✓ Model successfully saved and loaded!")


def example_5_batch_processing():
    """
    Example 5: Process multiple emails from a file
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Batch Email Processing")
    print("=" * 60)

    from spam_classifier import load_dataset
    from sklearn.model_selection import train_test_split

    # Train model
    df = load_dataset()
    X_train, _, y_train, _ = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    classifier = SpamClassifier()
    classifier.train(X_train, y_train)

    # Simulate batch of emails
    email_batch = [
        {"id": 1, "text": "Meeting at 3pm in conference room"},
        {"id": 2, "text": "Win $1000000 now! Click here!"},
        {"id": 3, "text": "Please review the attached invoice"},
        {"id": 4, "text": "Get viagra cheap! No prescription!"},
        {"id": 5, "text": "Your package has been delivered"}
    ]

    # Process batch
    texts = [email['text'] for email in email_batch]
    predictions = classifier.predict(texts)

    # Create results
    results = []
    for email, pred in zip(email_batch, predictions):
        results.append({
            'id': email['id'],
            'text': email['text'],
            'classification': 'spam' if pred == 1 else 'ham'
        })

    # Display results
    print("\nBatch Processing Results:")
    print("-" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Summary statistics
    spam_count = sum(predictions)
    ham_count = len(predictions) - spam_count
    print("-" * 60)
    print(f"Summary: {spam_count} spam, {ham_count} ham out of {len(predictions)} emails")


def example_6_error_handling():
    """
    Example 6: Proper error handling
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Error Handling")
    print("=" * 60)

    classifier = SpamClassifier()

    # Try to predict without training
    print("\nAttempting to predict without training...")
    try:
        classifier.predict(["Test email"])
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    # Train the model
    print("\nTraining model...")
    from spam_classifier import load_dataset
    from sklearn.model_selection import train_test_split

    df = load_dataset()
    X_train, _, y_train, _ = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    classifier.train(X_train, y_train)

    # Now predictions work
    print("Making predictions...")
    result = classifier.predict(["Test email"])
    print(f"✓ Prediction successful: {result[0]}")

    # Handle empty input
    print("\nTesting with empty input...")
    empty_result = classifier.predict([""])
    print(f"✓ Empty input handled: {empty_result[0]}")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 60)
    print("SPAM CLASSIFIER - EXAMPLE USAGE DEMONSTRATIONS")
    print("=" * 60)

    # Run examples
    example_1_quick_start()
    example_2_custom_emails()
    example_3_compare_algorithms()
    example_4_save_load_model()
    example_5_batch_processing()
    example_6_error_handling()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()