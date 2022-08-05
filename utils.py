"""
Utility Functions
"""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

from config import CFG
from dataset import create_dataloader
from models import create_model, create_tokenizer
from metrics import get_f1_score, get_r2_score, get_score

def sigmoid(x):
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

def get_prediction(outputs:np.array, threshold=0.5)->np.array:
    """get prediction from logits

    Args:
        outputs (np.array): outputs from model, shape: [1, 30]
        threshold (float, optional): sigmoid threshold. Defaults to 0.5.

    Returns:
        np.array: predictions
    """
    outputs = sigmoid(outputs)
    result = [0, 0, 0, 0, 0, 0]
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