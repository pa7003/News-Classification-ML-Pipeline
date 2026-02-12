from src.feature_engineering import analyze_text_features
from src.data_preprocessing import load_and_preprocess
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("Loading and preprocessing data...")
    train_df, test_df = load_and_preprocess()

    print("Analyzing text features...")
    analyze_text_features(train_df["text"])

    X_train = train_df["text"]
    X_test = test_df["text"]

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("\nTraining model with Pipeline...")
    train_model(X_train, y_train)

    print("\nEvaluating model...")
    accuracy = evaluate_model(X_test, y_test)

    print(f"\nFinal Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
