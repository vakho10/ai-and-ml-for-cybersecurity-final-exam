# Spam Email Classifier - User Guide

A machine learning-based system for automatically distinguishing between spam and legitimate (ham) emails using Natural Language Processing and classification algorithms.

---

## 📋 Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Understanding the Results](#understanding-the-results)
- [Customization Options](#customization-options)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This spam classifier uses machine learning to automatically identify spam emails. The system:
- **Preprocesses** email text to remove noise and standardize format
- **Extracts features** using TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifies** emails using a trained machine learning model
- **Achieves** high accuracy in distinguishing spam from legitimate emails

### Key Features
✅ Multiple algorithm support (Naive Bayes, Logistic Regression, SVM)  
✅ Automatic text preprocessing and cleaning  
✅ Performance visualization and metrics  
✅ Model persistence (save/load functionality)  
✅ Easy-to-use API for predictions

---

## 🔬 How It Works

### 1. **Text Preprocessing**
The classifier first cleans the email text:
- Converts text to lowercase
- Removes URLs, email addresses, and special characters
- Eliminates extra whitespace
- Standardizes the format

**Example:**
```
Input:  "GET VIAGRA NOW!!! Visit www.spam.com for BEST prices!!!"
Output: "get viagra now visit for best prices"
```

### 2. **Feature Extraction**
Uses TF-IDF to convert text into numerical features:
- **TF (Term Frequency)**: How often a word appears in an email
- **IDF (Inverse Document Frequency)**: How unique/important a word is across all emails
- Creates a 3000-dimensional feature vector for each email

### 3. **Classification**
The trained model analyzes the features and predicts:
- **0 (Ham)**: Legitimate email
- **1 (Spam)**: Spam email

### Visual Pipeline
```
Raw Email Text
      ↓
Text Preprocessing (cleaning, normalization)
      ↓
Feature Extraction (TF-IDF vectorization)
      ↓
Machine Learning Model (Naive Bayes/Logistic/SVM)
      ↓
Prediction: SPAM or HAM
```

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Quick Start
1. Clone or download the `task_3` folder
2. Navigate to the directory:
   ```bash
   cd task_3
   ```
3. Run the classifier:
   ```bash
   python spam_classifier.py
   ```

---

## 📖 Usage Guide

### Basic Usage

#### 1. Training a New Model
```python
from spam_classifier import SpamClassifier, load_dataset
from sklearn.model_selection import train_test_split

# Load your data
df = load_dataset()  # Or load from CSV: pd.read_csv('your_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Create and train classifier
classifier = SpamClassifier(model_type='naive_bayes')
classifier.train(X_train, y_train)

# Evaluate
metrics, predictions = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

#### 2. Making Predictions
```python
# Single email
emails = ["Win a FREE iPhone now! Click here!!!"]
predictions = classifier.predict(emails)
print("SPAM" if predictions[0] == 1 else "HAM")

# Multiple emails
emails = [
    "Meeting scheduled for tomorrow at 2pm",
    "You've won $1,000,000! Claim now!",
    "Please review the attached document"
]
predictions = classifier.predict(emails)
for email, pred in zip(emails, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{email[:40]}... → {label}")
```

#### 3. Saving and Loading Models
```python
# Save trained model
classifier.save_model('my_spam_model.pkl')

# Load model later
new_classifier = SpamClassifier()
new_classifier.load_model('my_spam_model.pkl')

# Use loaded model
result = new_classifier.predict(["Your email text here"])
```

### Using Your Own Dataset

To use your own email dataset:

```python
import pandas as pd

# Your CSV should have columns: 'text' (email content) and 'label' (0=ham, 1=spam)
df = pd.read_csv('my_emails.csv')

# Proceed with training
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)
classifier = SpamClassifier()
classifier.train(X_train, y_train)
```

**CSV Format Example:**
```csv
text,label
"Meeting tomorrow at 3pm",0
"You won $1000000! Click here!",1
"Please review the document",0
```

---

## 📊 Model Performance

### Performance Metrics Explained

When you run the classifier, you'll see these metrics:

| Metric | Description | What it Means |
|--------|-------------|---------------|
| **Accuracy** | Overall correctness | % of emails classified correctly |
| **Precision** | Spam detection accuracy | Of emails marked spam, how many truly are spam |
| **Recall** | Spam catch rate | Of all spam emails, how many were caught |
| **F1-Score** | Balance of precision & recall | Harmonic mean of precision and recall |

### Sample Results

```
Results:
├─ Accuracy:  0.9650
├─ Precision: 0.9545
├─ Recall:    0.9773
└─ F1-Score:  0.9658
```

**Interpretation:**
- **96.5%** of all emails are classified correctly
- **95.5%** of emails marked as spam are actually spam (low false positives)
- **97.7%** of actual spam is caught (low false negatives)

### Confusion Matrix

After training, you'll see a confusion matrix visualization saved as `confusion_matrix.png`:

```
                 Predicted
              Ham    Spam
Actual  Ham   [95]   [5]
        Spam  [2]    [98]
```

**How to read it:**
- **Top-left (95)**: Correctly identified ham emails
- **Bottom-right (98)**: Correctly identified spam emails
- **Top-right (5)**: Ham emails incorrectly marked as spam (False Positives)
- **Bottom-left (2)**: Spam emails that slipped through (False Negatives)

### Performance Visualization

The `metrics.png` file shows a bar chart of all performance metrics, making it easy to see how well the model performs at a glance.

---

## 🎨 Understanding the Results

### What Makes an Email Spam?

The model learns patterns commonly found in spam:
- **Urgency words**: "now", "urgent", "limited time"
- **Financial terms**: "free", "money", "win", "prize"
- **Call-to-action**: "click here", "order now", "buy"
- **Excessive punctuation**: "!!!", "???", "$$$"

### Common Spam Indicators
- Prize notifications
- Unsolicited offers
- Requests for personal information
- Suspicious links or attachments
- Poor grammar and spelling

### Common Ham Indicators
- Professional language
- Contextual relevance
- Proper formatting
- Known sender patterns
- Conversational tone

---

## ⚙️ Customization Options

### 1. Choosing Different Algorithms

```python
# Naive Bayes (Default - Fast, good for text)
classifier = SpamClassifier(model_type='naive_bayes')

# Logistic Regression (Balanced performance)
classifier = SpamClassifier(model_type='logistic')

# Support Vector Machine (Potentially higher accuracy)
classifier = SpamClassifier(model_type='svm')
```

### 2. Adjusting Feature Extraction

Modify the `TfidfVectorizer` parameters in the `__init__` method:

```python
# Increase number of features
self.vectorizer = TfidfVectorizer(
    max_features=5000,  # More features (default: 3000)
    stop_words='english',
    ngram_range=(1, 2)  # Include bigrams
)
```

### 3. Custom Preprocessing

Add custom preprocessing rules in the `preprocess_text` method:

```python
def preprocess_text(self, text):
    text = text.lower()
    # Add your custom rules here
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text
```

---

## 🔧 Troubleshooting

### Issue: Low Accuracy

**Solutions:**
1. Increase training data size
2. Try different algorithms
3. Adjust TF-IDF parameters (max_features)
4. Check data quality and balance

### Issue: Model Not Loading

**Solutions:**
1. Ensure the `.pkl` file exists
2. Check file path is correct
3. Verify pickle file is not corrupted
4. Retrain and save a new model

### Issue: Slow Predictions

**Solutions:**
1. Use Naive Bayes (fastest algorithm)
2. Reduce `max_features` in TfidfVectorizer
3. Process emails in batches

### Issue: Import Errors

```bash
# Install missing packages
pip install pandas numpy scikit-learn matplotlib seaborn

# Or install all at once
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
task_3/
├── spam_classifier.py          # Main implementation
├── spam_classifier_model.pkl   # Saved model (after training)
├── confusion_matrix.png        # Performance visualization
├── metrics.png                 # Metrics bar chart
└── README.md                   # This documentation
```

---

## 🎓 For Students

### Key Concepts Demonstrated

1. **Machine Learning Pipeline**: Data → Preprocessing → Feature Extraction → Model → Evaluation
2. **Text Processing**: Cleaning, normalization, vectorization
3. **Classification Algorithms**: Naive Bayes, Logistic Regression, SVM
4. **Model Evaluation**: Accuracy, precision, recall, F1-score
5. **Model Persistence**: Saving and loading trained models

### Understanding Cybersecurity Context

Spam filtering is crucial for cybersecurity because:
- **Phishing Prevention**: Many phishing attacks arrive via spam
- **Malware Distribution**: Spam is a common malware delivery vector
- **Social Engineering**: Spam often uses manipulation tactics
- **Resource Protection**: Reduces noise and protects productivity

---

## 📝 Notes

- The model works best with substantial training data (1000+ emails recommended)
- Balance your dataset (similar amounts of spam and ham) for optimal performance
- Regularly retrain with new data to adapt to evolving spam techniques
- Consider using the model as part of a layered security approach

---

## 🤝 Support

For issues or questions:
1. Review the troubleshooting section
2. Check that all dependencies are installed
3. Verify your data format matches the expected structure
4. Ensure Python version is 3.7 or higher

---

**Last Updated**: November 2025  
**Python Version**: 3.7+  
**License**: Educational Use