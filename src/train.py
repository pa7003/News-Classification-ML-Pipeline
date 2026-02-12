import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from src.config import MODEL_PATH, MAX_FEATURES, NGRAM_RANGE


def train_model(X_train, y_train):

    # Create Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            stop_words="english"
        )),
        ("classifier", LogisticRegression())
    ])

    # Hyperparameter grid (for classifier only)
    param_grid = {
        "classifier__C": [0.5, 1, 2],
        "classifier__max_iter": [1000, 2000],
        "classifier__solver": ["lbfgs"]
    }

    # GridSearch on entire pipeline
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best Parameters:", grid.best_params_)

    # Cross-validation on full pipeline
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=3,
        scoring="f1_weighted"
    )

    print("Cross-validation F1 scores:", cv_scores)
    print("Mean CV F1:", cv_scores.mean())

    # Save full pipeline
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    return best_model
