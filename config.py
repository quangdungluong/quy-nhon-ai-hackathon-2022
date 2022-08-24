"""
Config hyperparameters
"""
import torch
from transformers import (PhobertTokenizer, RobertaModel,
                          XLMRobertaModel, XLMRobertaTokenizer)


class CFG:
    seed = 2022
    num_labels = 6*5 # 6 aspects and 5 ratings for each
    num_folds = 5
    train_folds = [0, 1, 2, 3, 4]
    batch_size = 4
    scheduler_type = "cosine" # ["linear", "cosine"]
    num_cycles = 0.5
    num_warmup_steps = 100
    lr = 1e-5
    weight_decay = 0.01
    is_llrd = False
    llrd_ratio = 0.9
    num_epochs = 15
    max_len = 256
    hidden_dropout_prob = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### for multi-sample drop out
    num_drop = 5
    increment_dropout_prob = 0.05

    ### for label smoothing
    smoothing = [0.6, 0.2, 0.1, 0.05]
    is_smoothing = True

    ## For select model name and model type
    model_dict = {"xlm-roberta-base": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-base"),
                "xlm-roberta-large": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-large"),
                "phobert-base": (RobertaModel, PhobertTokenizer, "vinai/phobert-base"),
                "phobert-large": (RobertaModel, PhobertTokenizer, "vinai/phobert-large")}
