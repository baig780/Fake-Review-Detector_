import streamlit as st

# ✅ Fix: Move set_page_config to the top
st.set_page_config(page_title="Fake Review Detector", page_icon="📝", layout="centered")

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# ✅ Fix: Define NLTK Data Directory
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)
nltk.data.path.append(NLTK_DIR)

# ✅ Fix: Ensure NLTK Data is Available Before Running
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

# ✅ Fix: Custom Tokenizer to Avoid `punkt_tab` Error
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def custom_word_tokenize(text):
    return tokenizer.tokenize(text)

# ✅ Fix: Ensure Stopwords Are Loaded Properly
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DIR)
    stop_words = set(stopwords.words("english"))

# ✅ Load trained models and vectorizer safely
model_options = {
    "Logistic Regression": "fake_review_detector.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl"
}

try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    current_model_name = "Logistic Regression"
    model = joblib.load(model_options[current_model_name])
except FileNotFoundError:
    st.error("❌ Model files not found. Please upload the correct model files to your project directory.")

# ✅ Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = custom_word_tokenize(text)  # ✅ Use Custom Tokenizer
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Function to analyze sentiment
def analyze_sentiment(prob):
    if prob > 0.7:
        return "😃 Positive"
    elif 0.4 <= prob <= 0.7:
        return "😐 Neutral"
    else:
        return "😠 Negative"

# ✅ Set Streamlit page config
st.set_page_config(page_title="Fake Review Detector", page_icon="📝", layout="centered")

# ✅ App Title
st.markdown("<h1 class='title'>📝 Fake Review Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🚀 Made by <b>Abdul Rahman Baig</b></h4>", unsafe_allow_html=True)

# ✅ Dark Mode Toggle
dark_mode = st.checkbox("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #222; color: white; }</style>", unsafe_allow_html=True)

# ✅ Model Selection
selected_model = st.selectbox("Select a Model:", list(model_options.keys()))
if selected_model != current_model_name:
    try:
        model = joblib.load(model_options[selected_model])
        current_model_name = selected_model
    except FileNotFoundError:
        st.error(f"❌ {selected_model} model file not found. Please upload the correct file.")

# ✅ User Input Section
st.markdown("### 🔍 Enter a Review to Analyze")
user_review = st.text_area("✍️ Type your review here:")

if st.button("🚀 Analyze Review Now"):
    if user_review.strip():
        try:
            cleaned_review = clean_text(user_review)
            transformed_review = vectorizer.transform([cleaned_review])
            prediction = model.predict(transformed_review)[0]
            prob = model.predict_proba(transformed_review)[0]
            confidence = round(max(prob) * 100, 2)

            st.markdown("---")
            sentiment = analyze_sentiment(prob[1])

            if prediction == 1:
                st.markdown(f"<div class='review-box'>❌ **Fake Review Detected!** 😡 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='review-box'>✅ **Real Review!** 🎉 (Confidence: {confidence}%)</div>", unsafe_allow_html=True)

            # ✅ Confidence Score Visualization
            fig, ax = plt.subplots()
            ax.bar(["Real Review", "Fake Review"], prob * 100, color=["green", "red"])
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence Levels")
            st.pyplot(fig)

            # ✅ Option to download result
            result_text = f"Review: {user_review}\nPrediction: {'Fake Review' if prediction == 1 else 'Real Review'}\nConfidence: {confidence}%\nSentiment: {sentiment}"
            st.download_button(label="📥 Download Result", data=result_text, file_name="review_result.txt", mime="text/plain")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# ✅ Review Submission Section
st.markdown("---")  
st.subheader("📝 Give Your Honest Review About This App")  

reviewer_name = st.text_input("Your Name", "")
app_review = st.text_area("Your Review", "")

if st.button("Submit Review"):
    if reviewer_name.strip() and app_review.strip():
        review_entry = {"name": reviewer_name, "review": app_review}

        # ✅ Load existing reviews
        if os.path.exists("app_reviews.json"):
            with open("app_reviews.json", "r") as f:
                try:
                    review_data = json.load(f)
                except json.JSONDecodeError:
                    review_data = []
        else:
            review_data = []

        review_data.append(review_entry)

        # ✅ Save updated reviews
        with open("app_reviews.json", "w") as f:
            json.dump(review_data, f, indent=4)

        st.success("✅ Thank you for your feedback!")
    else:
        st.warning("⚠️ Please enter your name and review before submitting.")

# ✅ Display All User Reviews
st.markdown("---")  
st.subheader("📢 User Reviews About This App")

try:
    with open("app_reviews.json", "r") as f:
        review_data = json.load(f)

    if review_data:
        for review in review_data[-10:]:  # Show the last 10 reviews
            st.write(f"📝 **{review['name']}**: {review['review']}")
    else:
        st.info("No reviews yet. Be the first to leave feedback! 😊")
except FileNotFoundError:
    st.info("No reviews yet. Be the first to leave feedback! 😊")

st.markdown("---")

