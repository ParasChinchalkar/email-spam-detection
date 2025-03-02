import streamlit as st
import pickle
import os
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english')]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# File paths
vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")
model_path = os.path.join(os.getcwd(), "model.pkl")

# Function to train and save the model (if needed)
def train_and_save_model():
    emails = [
        "Congratulations! You've won a free vacation. Claim now!",  # Spam
        "Meeting at 5 PM. Please confirm attendance.",  # Not Spam
        "You have been selected for a prize! Click the link.",  # Spam
        "Can we reschedule our call for tomorrow?",  # Not Spam
    ]
    labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

    # Convert text to numerical features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(emails)
    y = np.array(labels)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)  # ✅ Training the model

    # Save vectorizer and model
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved successfully!")

# Check if model and vectorizer exist; if not, train
if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    st.warning("⚠️ Model or vectorizer not found. Training a new one...")
    train_and_save_model()

# Load the trained model and vectorizer
with open(vectorizer_path, "rb") as f:
    tfidf = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # 1. Preprocess the input
        transformed_sms = transform_text(input_sms)
        
        # 2. Convert text to numerical vector using TF-IDF
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict using the trained model
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")