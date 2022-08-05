import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import sigmoid

def get_r2_score(y_true: np.array, y_pred: np.array) -> float:
    """Calculate r2 score

    Args:
        y_true (np.array): ground truth
        y_pred (np.array): predictions

    Returns:
        float: r2 score
    """
    assert len(y_true) == len(y_pred)
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
    """Calculate precision, recall and f1 score

    Args:
        labels (np.array): grounth truth
        predictions (np.array): predictions

    Returns:
        tuple: (precision, recall, f1_score)
    """
    labels_ = labels > 0
    predictions_ = predictions > 0
    return (precision_score(labels_, predictions_), recall_score(labels_, predictions_), f1_score(labels_, predictions_))

def get_label(y_true: np.array) -> np.array:
    labels = np.array([0, 0, 0, 0, 0, 0])
    for i in range(6):
        for j in range(5):
            if y_true[5*i+j] == 1:
                labels[i] = j + 1
    return labels

def get_prediction(outputs:np.array, threshold=0.5)->np.array:
    """get prediction from logits

    Args:
        outputs (np.array): outputs from model, shape: [1, 30]
        threshold (float, optional): sigmoid threshold. Defaults to 0.5.

    Returns:
        np.array: predictions
    """
    outputs = sigmoid(outputs)
    result = np.array([0, 0, 0, 0, 0, 0])
    for i in range(6):
        best_score = -999
        index = -1
        for j in range(5):
            if outputs[5*i+j] > best_score:
                best_score = outputs[5*i+j]
                index = j
        if best_score > threshold:
            result[i] = index + 1
    return result

def report_score(labels, predictions) -> dict:
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
        dict: score of each aspect
    """
    score = {}
    preds = []
    labels = []
    for i in range(len(y_true)):
        preds.append(get_prediction(y_pred[i]))
        labels.append(get_label(y_true[i]))
    preds = np.array(preds)
    labels = np.array(labels)
    # np.save("preds.npy", preds)
    # np.save("labels.npy", labels)
    aspects = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    for i in range(6):
        score[aspects[i]] = report_score(preds[:,i], labels[:,i])
    return score