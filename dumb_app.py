import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load the saved model and vectorizer
model = load('checkpoint/best_model.pkl')
vectorizer = load('checkpoint/vectorizer.pkl')  # Assuming TF-IDF vectorizer was saved too
label_encoder = load('checkpoint/label_encoder.pkl')  # Assuming LabelEncoder was saved

# Function to preprocess input text
def preprocess_input(text):
    # Preprocess the input text (this should match the training process)
    text = text.lower()  # Convert to lowercase
    # Remove URLs, special characters, etc.
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and stem
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    tokens_stemmed = ' '.join([stemmer.stem(token) for token in tokens])
    return tokens_stemmed

# Function to classify the input and return results
def classify_text(input_text):
    start_time = time.time()
    
    # Preprocess and vectorize the input text
    processed_text = preprocess_input(input_text)
    text_tfidf = vectorizer.transform([processed_text])
    
    # Perform prediction
    prediction = model.predict(text_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Calculate execution time
    execution_time = time.time() - start_time
    
    return predicted_label, execution_time

# Streamlit App UI
st.title('Emotion Classification App')
st.write("This app classifies the emotion in a text and provides model performance metrics.")

# Input text box
input_text = st.text_area("Enter your text here:")

if st.button("Classify"):
    if input_text:
        # Get the classification result and execution time
        predicted_label, exec_time = classify_text(input_text)
        
        # Display the result
        st.write(f"### Emotion: {predicted_label}")
        st.write(f"### Execution Time: {exec_time:.4f} seconds")

        # Show model accuracy and confusion matrix
        st.subheader("Model Performance on Test Data")
        
        # Load test data for performance metrics (or you can use a cached version)
        df_test = pd.read_csv('data/dataset1.csv')  # Assuming you have test dataset

        # Preprocess test data
        X_test = df_test['statement'].apply(preprocess_input)
        X_test_tfidf = vectorizer.transform(X_test)
        y_test = label_encoder.transform(df_test['status'])
        
        # Predict on test data
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy on Test Data:** {accuracy:.4f}")
        
        # Display confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        labels = label_encoder.classes_

        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

        # Show classification report
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred, target_names=labels))

    else:
        st.write("Please enter some text for classification.")
