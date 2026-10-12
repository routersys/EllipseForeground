import numpy as np
from scipy.stats import rankdata


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))
