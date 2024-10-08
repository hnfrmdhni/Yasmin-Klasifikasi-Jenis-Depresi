import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import time
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model, TF-IDF Vectorizer, and LabelEncoder
with open('checkpoint/kaggle_n_estimators500_max_length50000_GPU_HIST/XGB_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('checkpoint/kaggle_n_estimators500_max_length50000_GPU_HIST/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('checkpoint/kaggle_n_estimators500_max_length50000_GPU_HIST/label_encoder.pkl', 'rb') as label_file:
    lbl_enc = pickle.load(label_file)

# Preprocessing function with additional check for non-string inputs
def preprocess_text(text):
    # Check if the input is a string, if not, return an empty string
    if not isinstance(text, str):
        return ""
    
    # Lowercasing
    text = text.lower()
    
    # Remove patterns (URLs, handles, special characters)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Stem tokens
    stemmer = PorterStemmer()
    stemmed_tokens = ' '.join([stemmer.stem(token) for token in tokens])
    
    return stemmed_tokens

# Streamlit interface
st.title("Depression Type Classifier")

st.write("Input a text to classify the type of depression based on your input.")

# User input
user_input = st.text_area("Enter your text here:")

if st.button("Classify"):
    if user_input:
        # Start time measurement
        start_time = time.time()
        
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Convert the processed text to TF-IDF features
        input_tfidf = vectorizer.transform([processed_text])
        
        # Add additional numerical features (if any), here it's empty
        num_features = [[len(user_input), len(nltk.sent_tokenize(user_input))]]
        
        # Combine text features and numerical features
        input_combined = hstack([input_tfidf, num_features])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_combined)
        
        # Decode the prediction using LabelEncoder
        predicted_label = lbl_enc.inverse_transform(prediction)[0]
        
        # End time measurement
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        # Display the result
        st.write(f"**Prediction:** {predicted_label.capitalize()}")
        st.write(f"### Execution Time: {execution_time:.4f} seconds")
    else:
        st.write("Please enter a text to classify.")

# Button to show model performance on test data
if st.button("Show Performance"):
    st.subheader("Model Performance on Test Data")
    
    # Load test data for performance metrics (or you can use a cached version)
    df_test = pd.read_csv('data/dataset1.csv')  # Assuming you have test dataset

    # Preprocess the 'statement' column, filling NaN values with an empty string
    df_test['statement'] = df_test['statement'].fillna("")
    
    # Apply the preprocess_text function
    X_test = df_test['statement'].apply(preprocess_text)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Add additional numerical features (assuming the same used during training)
    num_features_test = [[len(text), len(nltk.sent_tokenize(text))] for text in df_test['statement']]
    
    # Combine text features and numerical features
    X_test_combined = hstack([X_test_tfidf, num_features_test])
    
    # Transform labels
    y_test = lbl_enc.transform(df_test['status'])
    
    # Predict on test data
    y_pred = model.predict(X_test_combined)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy on Test Data:** {accuracy:.4f}")
    
    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = lbl_enc.classes_

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