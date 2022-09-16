"""
Config hyperparameters
"""
import torch
from transformers import (PhobertTokenizer, RobertaModel, RobertaTokenizer,
                          XLMRobertaModel, XLMRobertaTokenizer)

model_name = "phobert-base"
model_type = "dropcat"
model_ckpt = "model.bin"
rdrsegmenter_path = "VnCoreNLP/VnCoreNLP-1.1.1.jar"
num_labels = 6*5 # 6 aspects and 5 ratings for each
batch_size = 4
max_len = 256
hidden_dropout_prob = 0.1
device = torch.device("cpu")

### for multi-sample drop out
num_drop = 5
increment_dropout_prob = 0.1

## For select model name and model type
model_dict = {"xlm-roberta-base": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-base"),
            "xlm-roberta-large": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-large"),
            "phobert-base": (RobertaModel, PhobertTokenizer, "vinai/phobert-base"),
            "phobert-large": (RobertaModel, PhobertTokenizer, "vinai/phobert-large")}
