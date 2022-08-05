"""
Config hyperparameters
"""
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import PhobertTokenizer

class CFG:
    seed = 42
    num_labels = 6*5
    num_folds = 5
    train_folds = [0]
    batch_size = 4
    lr = 1e-5
    weight_decay = 0.01
    num_epochs = 10
    max_len = 512
    hidden_dropout_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### for multi-sample drop out
    num_drop = 5
    increment_dropout_prob = 0.01

    ## For select model name and model type
    model_dict = {"xlm-roberta-base": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-base"),
                "xlm-roberta-large": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-large"),
                "phobert-base": (RobertaModel, PhobertTokenizer, "vinai/phobert-base"),
                "phobert-large": (RobertaModel, PhobertTokenizer, "vinai/phobert-large")}