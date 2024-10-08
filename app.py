import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Load the trained model, TF-IDF Vectorizer, and LabelEncoder
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as label_file:
    lbl_enc = pickle.load(label_file)

# Preprocessing function
def preprocess_text(text):
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
        
        # Display the result
        st.write(f"**Prediction:** {predicted_label.capitalize()}")
        
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
        st.write("Please enter a text to classify.")
