import numpy as np


def analyze_text_features(text_series):
    """
    Perform basic text feature analysis.
    Returns statistics useful for understanding dataset.
    """

    text_lengths = text_series.apply(lambda x: len(x.split()))

    stats = {
        "Average Length": np.mean(text_lengths),
        "Max Length": np.max(text_lengths),
        "Min Length": np.min(text_lengths)
    }

    print("\nText Feature Analysis:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats
