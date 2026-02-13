import streamlit as st
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.path.join("models", "news_classifier.pkl")
TRAIN_PATH = os.path.join("data", "raw", "train.csv")
TEST_PATH = os.path.join("data", "raw", "test.csv")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📰 News Article Classification App")
st.write("Enter a news article text below to classify it into a category.")

# ----------------------------
# PREDICTION SECTION
# ----------------------------
user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input.strip():
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

# ----------------------------
# EVALUATION SECTION
# ----------------------------
st.markdown("---")
st.header("📊 Model Evaluation Metrics (Test Dataset)")

@st.cache_data
def load_test_data():
    test_df = pd.read_csv(TEST_PATH)
    test_df.columns = ["label", "title", "description"]
    test_df["text"] = test_df["title"] + " " + test_df["description"]
    return test_df[["text", "label"]]

if st.button("Run Evaluation on Test Set"):

    test_df = load_test_data()

    X_test = test_df["text"]
    y_test = test_df["label"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    st.subheader("Overall Metrics")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision (Weighted):** {precision:.4f}")
    st.write(f"**Recall (Weighted):** {recall:.4f}")
    st.write(f"**F1 Score (Weighted):** {f1:.4f}")

    # Classification Report
    report = classification_report(y_test, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
