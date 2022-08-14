"""
Utility Functions
"""
import os
import random

import numpy as np
import torch
from tqdm import tqdm

def sigmoid(x:np.array) -> np.array:
    """sigmoid function

    Args:
        x (np.array): x

    Returns:
        np.array: sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))

def seed_everything(seed: int):
    """seed everything

    Args:
        seed (int): hash seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_label(y_true: np.array) -> np.array:
    """Convert from one-hot vector into multi-labels vector

    Args:
        y_true (np.array): the one-hot vector of ground truth labels, shape [1,30]

    Returns:
        np.array: ground truth, multi-labels vector, shape [1,6]
    """
    labels = np.array([0, 0, 0, 0, 0, 0])
    for i in range(6):
        for j in range(5):
            if y_true[5*i+j] == 1:
                labels[i] = j + 1
    return labels

def get_prediction(outputs:np.array, threshold=0.5)->np.array:
    """get prediction from logits \\
    Convert from logits into multi-labels vector

    Args:
        outputs (np.array): outputs logits from model, shape: [1, 30]
        threshold (float, optional): sigmoid threshold. Defaults to 0.5.

    Returns:
        np.array: predictions, multi-labels vector, shape: [1,6]
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
        if best_score >= threshold:
            result[i] = index + 1
    return result

def get_final_prediction(logits:list, threshold=0.5):
    """get final prediction

    Args:
        logits (list): a list of outputs model, before go through sigmoid
        threshold (float, optional): threshold. Defaults to 0.5.

    Returns:
        predictions: a list of predictions of 6 aspects
        probs: a list of dictionaries of probabilities after sigmoid of 6 aspects and 5 rating
    """
    predictions = []
    probs = {"giai_tri": [],
             "luu_tru": [],
             "nha_hang": [],
             "an_uong": [],
             "di_chuyen": [],
             "mua_sam": []}
    names = ["giai_tri","luu_tru","nha_hang","an_uong","di_chuyen","mua_sam"]
    for logit in logits:
        predictions.append(get_prediction(logit, threshold))
        logit = sigmoid(logit)
        for i in range(6):
            prob = [logit[i*5+j] for j in range(5)]
            probs[names[i]].append(prob)
    return predictions, probs