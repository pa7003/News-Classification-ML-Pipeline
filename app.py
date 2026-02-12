import streamlit as st
import pickle

MODEL_PATH = "models/news_classifier.pkl"

st.title("📰 News Article Classification App")

st.write("Enter a news article text below to classify it into a category.")

# Load model
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]

        category_map = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech"
        }

        st.success(f"Predicted Category: {category_map.get(prediction, prediction)}")
    else:
        st.warning("Please enter some text.")
