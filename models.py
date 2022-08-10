import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

from config import CFG


def get_dropouts(num:int, start_prob:float, increment:float) -> nn.Module:
    """Get multiple dropouts

    Args:
        num (int): number of dropout layers
        start_prob (float): start dropout probability
        increment (float): dropout probability increment

    Returns:
        nn.Module: multiple dropouts
    """
    
    return [StableDropout(start_prob + (increment * i)) for i in range(num)]

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class CustomModel(nn.Module):
    """Custom model
    return:
    outputs: torch.Size([batch_size, num_labels])
    """
    def __init__(self, model, model_ckpt) -> None:
        super().__init__()
        self.model = model.from_pretrained(model_ckpt)
        self.pooling = MeanPooling()
        self.dropout = nn.Dropout(p=CFG.hidden_dropout_prob)
        self.classifer = nn.Linear(self.model.config.hidden_size, CFG.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        outputs = self.pooling(outputs.last_hidden_state, attention_mask)
        outputs = self.dropout(outputs)
        outputs = self.classifer(outputs)
        return outputs


# Concat the last 4 hidden representations of the [CLS] token, and fed it into simple MLP
# Ref: https://github.com/suicao/PhoBert-Sentiment-Classification/blob/master/models.py
class HackathonModel(nn.Module):
    def __init__(self, model, model_ckpt) -> None:
        super().__init__()
        self.model = model.from_pretrained(model_ckpt)
        self.classifier = nn.Linear(4*self.model.config.hidden_size, CFG.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_output = torch.cat((outputs[2][-1][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
        outputs = self.classifier(cls_output)
        return outputs
    
class ClassifierHead(nn.Module):
    """
    Bert base with a Linear layer plopped on top of it
    - connects the CLS token of the last hidden layer with the FC
    """
    def __init__(self, model, model_ckpt) -> None:
        super().__init__()
        self.model = model.from_pretrained(model_ckpt)
        self.cnn = nn.Conv1d(self.model.config.hidden_size, CFG.num_labels, kernel_size=1)
        self.classifier = nn.Linear(self.model.config.hidden_size, CFG.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs[0]
        
        # Max-pool on a CNN of all tokens of the last hidden layer
        # hidden_states = hidden_states.permute(0, 2, 1)
        # cnn_states = self.cnn(hidden_states)
        # cnn_states = cnn_states.permute(0, 2, 1)
        # outputs, _ = torch.max(cnn_states, 1)
        
        # FC on 1st token
        outputs = self.classifier(hidden_states[:, 0, :])
        return outputs
    
class MeanMaxModel(nn.Module):
    """
    Concat mean and max pooling
    """
    def __init__(self, model, model_ckpt) -> None:
        super().__init__()
        self.model = model.from_pretrained(model_ckpt)
        self.dropout = nn.Dropout(CFG.hidden_dropout_prob)
        self.classifier = nn.Linear(2*self.model.config.hidden_size, CFG.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        avg_pool = torch.mean(outputs.last_hidden_state, 1)
        max_pool = torch.max(outputs.last_hidden_state, 1)
        x = torch.cat((avg_pool, max_pool), 1)
        x =  self.dropout(x)
        x = self.classifier(x)
        return x
    
class ElectraClassification(nn.Module):
    def __init__(self, model, model_ckpt) -> None:
        super().__init__()
        self.model = model.from_pretrained(model_ckpt)
        self.classifier = nn.Linear(4*self.model.config.hidden_size, CFG.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # outputs[1]
        cls_output = torch.cat((outputs[1][-1][:,0, ...],outputs[1][-2][:,0, ...], outputs[1][-3][:,0, ...], outputs[1][-4][:,0, ...]),-1)
        outputs = self.classifier(cls_output)
        return outputs

class MultiDropModel(nn.Module):
    def __init__(self, model, model_ckpt):
        super(MultiDropModel, self).__init__()

        self.bert = model.from_pretrained(model_ckpt)
        self.pooler = MeanPooling()
        self.drop = get_dropouts(num = CFG.num_drop, start_prob = CFG.hidden_dropout_prob, increment= CFG.increment_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, CFG.num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states=True)
        pooler_output = self.pooler(bert_output.last_hidden_state, attention_mask)

        num_dps = float(len(self.drop))

        for ii, drop in enumerate(self.drop):
            if ii == 0:
                outputs = (self.classifier(drop(pooler_output)) / num_dps)
            else :
                outputs += (self.classifier(drop(pooler_output)) / num_dps)
        
        return outputs

class DropCatModel(nn.Module):
    def __init__(self, model, model_ckpt):
        super(DropCatModel, self).__init__()

        self.model = model.from_pretrained(model_ckpt)
        self.drop = get_dropouts(num = CFG.num_drop, start_prob = CFG.hidden_dropout_prob, increment= CFG.increment_dropout_prob)
        self.classifier = nn.Linear(4 * self.bert.config.hidden_size, CFG.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
        cat_output = torch.cat((outputs[2][-1][:,0, ...], outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)

        num_dps = float(len(self.drop))
        for ii, drop in enumerate(self.drop):
            if ii == 0:
                outputs = (self.classifier(drop(cat_output)) / num_dps)
            else :
                outputs += (self.classifier(drop(cat_output)) / num_dps)
        return outputs

def create_model(model_name:str, model_type:str) -> nn.Module:
    """Create model

    Args:
        model_name (str): model name, args.model_name
        model_type (str): model type, choose head, args.model_type

    Returns:
        nn.Module: model
    """
    model, tokenizer, model_ckpt = CFG.model_dict[model_name]
    if model_type == "4_hidden":
        return HackathonModel(model, model_ckpt)
    elif model_type == "classifier_head":
        return ClassifierHead(model, model_ckpt)
    elif model_type == "mean_max":
        return MeanMaxModel(model, model_ckpt)
    elif model_type == "electra":
        return ElectraClassification(model, model_ckpt)
    elif model_type == "multi_drop":
        return MultiDropModel(model, model_ckpt)
    elif model_type == "4_hidden_drop":
        return DropCatModel(model, model_ckpt)
    return CustomModel(model, model_ckpt)


def create_tokenizer(name:str) -> Tokenizer:
    """Create tokenizer

    Args:
        name (str): model name, args.model
        model_ckpt (str): model checkpoint

    Returns:
        Tokenizer: tokenizer
    """
    model, tokenizer, model_ckpt = CFG.model_dict[name]
    return tokenizer.from_pretrained(model_ckpt)
