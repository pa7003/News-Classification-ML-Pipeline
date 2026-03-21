import pandas as pd
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.config import TRAIN_PATH, TEST_PATH
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()   # replace multiple spaces with one space

    words = text.split()       # tokenization 
    words = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]         # remove stopwards and short words
    words = [lemmatizer.lemmatize(word) for word in words]       # converts words to their root form 

    return " ".join(words)     # converts list to str


def load_and_preprocess():
    # AG News has no headers
    # Load dataset
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Rename columns
    train_df.columns = ["label", "title", "description"]
    test_df.columns = ["label", "title", "description"]

    # Combine title + description
    train_df["text"] = train_df["title"] + " " + train_df["description"]
    test_df["text"] = test_df["title"] + " " + test_df["description"]

    # Clean text
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    train_df = train_df[train_df["text"].str.strip() != ""]
    test_df = test_df[test_df["text"].str.strip() != ""]

    return train_df[["text", "label"]], test_df[["text", "label"]]
