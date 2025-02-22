import streamlit as st
import numpy as np
import joblib
from textblob import TextBlob
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import re

# Load pre-trained model and vectorizer
model = joblib.load("ada.pkl")  # Ensure the model is saved
bow_vectorizer = joblib.load("bow.pkl")  # Load BoW vectorizer
tfidf_vectorizer = joblib.load("tfidf.pkl")  # Load tfidf vectorizer

# for word tokenzier
nltk.download('punkt_tab')
nltk.download('wordnet')

# for POS
nltk.download('averaged_perceptron_tagger_eng')

# Add features
def add_features(message):
    data = pd.DataFrame({"textraw": [message]})

    # Feature 1: Message length
    data['message_length'] = data['textraw'].apply(len)

    # Feature 2: Number of special characters
    data['special_char_count'] = data['textraw'].apply(lambda x: sum(1 for c in x if c in string.punctuation))

    # Feature 3: Number of digits in a message
    data['digit_count'] = data['textraw'].apply(lambda x: sum(1 for c in x if c.isdigit()))

    # Feature 4: Contains URL (1 = Yes, 0 = No)
    data['contains_url'] = data['textraw'].apply(lambda x: 1 if re.search(r"http[s]?://|www\.", x) else 0)

    # Feature 5: Uppercase Word Count
    data['uppercase_word_count'] = data['textraw'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))

    # Feature 6: Exclamation Mark Count
    data['exclamation_count'] = data['textraw'].apply(lambda x: x.count("!"))

    # Feature 7: Word Count
    data['word_count'] = data['textraw'].apply(lambda x: len(x.split()))

    # Feature 8: Average Word Length
    data['avg_word_length'] = data['textraw'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)

    # Feature 9: Sentiment score
    data['Sentiment'] = [TextBlob(message).sentiment.polarity]

    # Feature 10:POS
    tokens = word_tokenize(message)
    pos_tags = pos_tag(tokens)

    data['Noun_count'] = len([word for word, tag in pos_tags if tag.startswith('NN')])  # Noun
    data['Verb_count'] = len([word for word, tag in pos_tags if tag.startswith('VB')])  # Verb
    data['Adjectives_count'] = len([word for word, tag in pos_tags if tag.startswith('JJ')])  # Adjective
    data['Adverbs_count'] = len([word for word, tag in pos_tags if tag.startswith('RB')])  # Adverb

    # tfidf vector
    tfidf = tfidf_vectorizer.transform([message])  # Convert to numerical form
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    data = pd.concat([data,tfidf_df], axis = 1)

    # bow vector
    bow = bow_vectorizer.transform([message])  # Convert to numerical form
    bow_df = pd.DataFrame(bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
    data = pd.concat([data,bow_df], axis = 1)

    data.drop('textraw', axis=1, inplace=True)
    return data

# Function to classify the input message
def classify_message(message, model):
    message = message.lower().strip()  # Preprocess message
    data = add_features(message)
    
    probabilities = model.predict_proba(data)[0]
    spam_prob, ham_prob = probabilities[1], probabilities[0]
    prediction = "Spam" if spam_prob > ham_prob else "Ham"

    return prediction, spam_prob, ham_prob

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check if it's **Spam or Ham**.")

# Input text box
user_message = st.text_area("Type your message here:")

# Predict button
if st.button("Check Message"):
    if user_message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        prediction, spam_prob, ham_prob = classify_message(user_message, model)
        
        # Display result
        st.subheader(f"Prediction: **{prediction}**")
        st.write(f"📊 **Spam Probability:** {spam_prob:.4f}")
        st.write(f"📊 **Ham Probability:** {ham_prob:.4f}")
        
        if prediction == "Spam":
            st.error("🚨 This message is classified as Spam!")
        else:
            st.success("✅ This message is classified as Ham.")

# About section
st.markdown("---")
st.write("🔍 **How It Works**")
st.write("""
- This app uses **Naïve Bayes with Bag of Words (BoW)** to classify messages.
- It analyzes the words in your message and assigns a probability of being **Spam or Ham**.
- If the model does not recognize any words, it may return **"Unknown (Out-of-Vocabulary)".**
""")
