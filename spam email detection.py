# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings for clean output

# Load the dataset and check columns
dataset = pd.read_csv("mail_data.csv", encoding='latin-1')
print("Original Columns:", dataset.columns)

# Automatically select the first two columns and rename them
dataset = dataset.iloc[:, :2]
dataset.columns = ['Category', 'Message']

# Convert 'Category' column into numerical labels (ham = 0, spam = 1)
dataset['Category'] = dataset['Category'].map({'ham': 0, 'spam': 1})

# Display first few rows
print("Dataset Preview:")
print(dataset.head())

# Count plot of spam vs. ham messages
sns.countplot(x='Category', data=dataset)
plt.title("Spam vs. Ham Messages Count")
plt.xlabel("Message Type (0: Ham, 1: Spam)")
plt.ylabel("Count")
plt.show()

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(dataset['Message'], dataset['Category'], test_size=0.2, random_state=0)

# Convert text into numerical features using CountVectorizer (Bag of Words)
cv = CountVectorizer()
X_train_transformed = cv.fit_transform(X_train)
X_test_transformed = cv.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Display Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test the model with a new message
def predict_spam(message):
    message_transformed = cv.transform([message])  # Transform message using trained CountVectorizer
    prediction = model.predict(message_transformed)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example messages to test
test_messages = [
    "Congratulations! You have won a free gift. Click here to claim now.",
    "Hey, are we still meeting for lunch today?",
    "You have been selected for a lottery prize worth $10,000! Claim now.",
    "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    "URGENT! Your account has been compromised. Click this link to secure it."
]

print("\nMessage Predictions:")
for msg in test_messages:
    print(f"Message: {msg} --> Prediction: {predict_spam(msg)}")
