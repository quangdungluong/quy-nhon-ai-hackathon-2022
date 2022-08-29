"""
Implement competition metrics
Sorry for this very silly, dummy code. Too much magic number :((
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import get_label, get_prediction


def get_r2_score(y_true: np.array, y_pred: np.array) -> float:
    """Calculate r2 score \\
    R2 = 1 - RSS/K \\
    RSS = residual sum of squares = sigma [(y_hat - y) ^ 2] \\
    K = total sum of squares of max distance = n * (max_sentiment - min_sentiment) ^ 2 \\

    Args:
        y_true (np.array): ground truth
        y_pred (np.array): predictions

    Returns:
        float: r2 score
    """
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape, y_true has shape {y_true.shape} while y_pred has shape {y_pred.shape}"
    y_true_ = y_true
    y_pred_ = y_pred
    max_sentiment = 5
    min_sentiment = 1
    rss = 0
    n = 0
    for i in range(len(y_true_)):
        if y_true_[i] * y_pred_[i] > 0:
            rss += (y_true_[i] - y_pred_[i])**2
            n += 1
    try :
        return 1 - rss/(n * (max_sentiment-min_sentiment)**2)
    except :
        return 1 # competition rules

def get_precision_recall_f1_score(labels:np.array, predictions:np.array) -> tuple:
    """Calculate precision, recall and f1 score \\
    [0] -> 0 and [1-5] -> 1

    Args:
        labels (np.array): grounth truth
        predictions (np.array): predictions

    Returns:
        tuple: (precision, recall, f1_score)
    """
    labels_ = labels > 0
    predictions_ = predictions > 0
    return (precision_score(labels_, predictions_), recall_score(labels_, predictions_), f1_score(labels_, predictions_))

def report_score(labels:np.array, predictions:np.array) -> dict:
    """Get the score: precision, recall, f1 score, r2 score and competition score \\
    Score of 1 aspect

    Args:
        labels (np.array): ground truth, shape: [num_examples, 1]
        predictions (np.array): predictions, shape: [num_examples, 1]

    Returns:
        dict: a score dictionary {precision, recall, f1_score, r2_score, competition_score} of 1 aspect
    """
    score = {}
    score["precision"], score["recall"], score["f1_score"] = get_precision_recall_f1_score(labels, predictions)
    score["r2_score"] = get_r2_score(labels, predictions)
    score["competition_score"] = score["f1_score"] * score["r2_score"]
    return score

def get_score(y_true: np.array, y_pred: np.array) -> dict:
    """Get score

    Args:
        y_true (np.array): shape: [n_examples, 30]
        y_pred (np.array): shape: [n_examples, 30], logits, before sigmoid

    Returns:
        dict: a score dictionary of each aspect
    """
    score = {}
    preds = []
    labels = []
    for i in range(len(y_true)):
        preds.append(get_prediction(y_pred[i]))
        labels.append(get_label(y_true[i]))
        
    preds = np.array(preds)
    labels = np.array(labels)
    aspects = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    for i in range(6):
        score[aspects[i]] = report_score(labels[:,i], preds[:,i])
    return score
