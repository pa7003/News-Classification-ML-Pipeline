---

# 📰 News Article Classification using Machine Learning

## 📌 Project Overview

This project implements a complete machine learning pipeline to classify news articles into predefined categories using traditional NLP techniques. The solution is built entirely using Python scripts (no notebooks) and follows a modular, production-style architecture. The pipeline includes preprocessing, feature engineering, hyperparameter tuning, cross-validation, and model evaluation.

---

## 📂 Dataset Source

The project uses the **AG News Dataset**, a widely used benchmark dataset for text classification.

* Source: AG News Dataset (Kaggle Version)
* Categories:

  * World
  * Sports
  * Business
  * Sci/Tech
* Training samples: 120,000
* Test samples: 7,600

Each record contains:

* Class Index (label)
* Title
* Description

Title and description are combined for text classification.

---

## 🏗️ Folder Structure Explanation

```
news_classification_project/
│
├── data/
│   ├── raw/              # Contains original train.csv and test.csv
│   └── processed/        # (Optional) Processed data files
│
├── src/
│   ├── config.py              # Configuration variables and file paths
│   ├── data_preprocessing.py  # Data loading and text cleaning
│   ├── feature_engineering.py # Text feature analysis utilities
│   ├── train.py               # Model training with Pipeline + GridSearchCV
│   └── evaluate.py            # Model evaluation and metrics saving
│
├── models/
│   └── news_classifier.pkl    # Saved trained Pipeline model
│
├── results/
│   └── metrics.txt            # Evaluation metrics output
│
├── requirements.txt
├── README.md
└── main.py                    # Entry point (runs full pipeline)
```

The project is fully modular and executable from the terminal using a single command.

---

## ▶️ Steps to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd news_classification_project
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Place Dataset Files

Ensure the following files are inside:

```
data/raw/
```

* train.csv
* test.csv

### 5️⃣ Run the Project

```bash
python main.py
```

The pipeline will:

* Load and preprocess data
* Perform text feature analysis
* Train model with hyperparameter tuning
* Evaluate performance
* Save trained model
* Save evaluation metrics

Final accuracy will be printed in the terminal.

---

## 🤖 Model Used

The project uses a **scikit-learn Pipeline** combining:

* **TF-IDF Vectorization**

  * max_features = 15,000
  * ngram_range = (1,2)
  * stopword removal

* **Logistic Regression (Multiclass)**

  * Hyperparameter tuning using GridSearchCV
  * 3-fold Cross Validation
  * Weighted F1 scoring for balanced evaluation

Why Logistic Regression?

* Performs strongly on high-dimensional sparse text data
* Efficient and interpretable
* Well-suited for multiclass classification problems

The final trained Pipeline (vectorizer + classifier) is saved as:

```
models/news_classifier.pkl
```

---

## 📊 Final Result Summary

* **Cross-Validation (3-fold) Mean F1 Score:** ≈ 0.899
* **Final Test Accuracy:** **91.55%**
* Evaluation metrics include:

  * Accuracy
  * Precision (weighted)
  * Recall (weighted)
  * F1-score (weighted)
  * Confusion Matrix
  * Full Classification Report

Results are saved in:

```
results/metrics.txt
```

The model demonstrates strong generalization and stable cross-validation performance on a balanced multiclass dataset.

---

## Streamlit Deployment
* Link : https://news-classification-7003.streamlit.app/
