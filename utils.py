"""
Utility Functions
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from config import CFG


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


def get_optimizer(model:nn.Module):
    """get optimizer

    Args:
        model (nn.Module): model

    Returns:
        optimizer
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=CFG.lr)
    return optimizer
    
def get_group_optimizer(model:nn.Module):
    # differential learning rate and weight decay
    learning_rate = CFG.lr
    no_decay = ['bias', 'gamma', 'beta']
    group1 = ['layer.0.','layer.1.','layer.2.','layer.3.']
    group2 = ['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3 = ['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all = ['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': CFG.weight_decay, 'lr': learning_rate / 2.6},
        {'params' : [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': CFG.weight_decay},
        {'params' : [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': CFG.weight_decay, 'lr': learning_rate},
        {'params' : [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': CFG.weight_decay, 'lr': learning_rate * 2.6},
        {'params' : [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
        {'params' : [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0, 'lr': learning_rate / 2.6},
        {'params' : [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0, 'lr': learning_rate},
        {'params' : [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0, 'lr': learning_rate * 2.6},
        {'params' : [p for n, p in model.named_parameters() if "model" not in n], 'lr' : CFG.lr, "momentum" : 0.99},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=CFG.lr)
    return optimizer

def get_scheduler(optimizer, scheduler_type:str, num_training_steps:int):
    """get scheduler

    Args:
        optimizer (_type_): optimizer
        scheduler_type (str): type of  scheduler
        num_training_steps (int): total training steps

    Returns:
        scheduler 
    """
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_training_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_training_steps, num_cycles=CFG.num_cycles)
    
    return scheduler    

def get_layerwise_lr_decay(model:nn.Module):
    """get layerwise learning rate decay

    Args:
        model (nn.Module): model

    Returns:
        optimizer_grouped_parameters : group optimizer parameters
    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": CFG.weight_decay,
            "lr": CFG.lr,
        },
    ]
    # initialize lrs for every layer
    layers = [getattr(model, 'bert').embeddings] + list(getattr(model, 'bert').encoder.layer)
    layers.reverse()
    lr = CFG.lr
    for layer in layers:
        lr *= CFG.llrd_ratio
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": CFG.weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters

def preprocess(text:str) -> str:
    """Replace acronym words

    Args:
        text (str): original text

    Returns:
        str: returned text
    """
    # replace_dict = {"Nvien" : "nh??n vi??n", "Nv": "nh??n vi??n", "NV":"nh??n vi??n", "nvien": "nh??n vi??n", "nv":"nh??n vi??n", "NVien":"nh??n vi??n",
    #                         "ks":"kh??ch s???n", "ksan":"kh??ch s???n", "Ks":"kh??ch s???n", "KS":"kh??ch s???n", "Ksan":"kh??ch s???n",
    #                         "cf":"c?? ph??", "Cf":"c?? ph??", "CF":"c?? ph??", "coffee":"c?? ph??",
    #                         "qn":"quy nh??n", "Qn":"quy nh??n", "QN":"quy nh??n",
    #                         "ch???": "ch???", "Ch???":"Ch???",
    #                         "bt":"b??nh th?????ng",
    #                         "l???u":"l???u",
    #                         "tp":"th??nh ph???", "TP":"th??nh ph???", "Tp":"th??nh ph???", "tpho":"th??nh ph???",
    #                         "nh??u":"nhi???u",
    #                         "mk":"m??nh",
    #                         "nc":"n?????c",
    #                         "u???g":"u???ng",
    #                         "mn":"m???i ng?????i", "mng":"m???i ng?????i", "Mn":"m???i ng?????i", "Mng":"m???i ng?????i"}
    replace_dict = {"Nvien" : "nh??n vi??n", "Nv": "nh??n vi??n", "NV":"nh??n vi??n", "nvien": "nh??n vi??n", "nv":"nh??n vi??n", "NVien":"nh??n vi??n",
                            "ks":"kh??ch s???n", "ksan":"kh??ch s???n", "Ks":"kh??ch s???n", "KS":"kh??ch s???n", "Ksan":"kh??ch s???n"}
    text_ls = text.split()
    for idx, word in enumerate(text_ls):
        if word in replace_dict:
            text_ls[idx] = replace_dict[word]
    text = " ".join(text_ls)
    return text
