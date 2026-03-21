import numpy as np


def analyze_text_features(text_series):
    """
    Perform basic text feature analysis.
    Returns statistics useful for understanding dataset.
    """

    text_lengths = text_series.apply(lambda x: len(x.split()))    # split sentence into words and then counts number of words

    stats = {
        "Average Length": np.mean(text_lengths),   # Average number of words per text
        "Max Length": np.max(text_lengths),        # Longest text in dataset
        "Min Length": np.min(text_lengths)         # Shortest text in dataset
    }

    print("\nText Feature Analysis:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats
