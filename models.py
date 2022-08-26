import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers = 12, layer_start = 4, layer_weights = None):
        super().__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super().__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(CFG.hidden_dropout_prob)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(CFG.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(CFG.device)

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

class AttentionModel(nn.Module):
    def __init__(self, model, model_ckpt, hiddendim_fc = 128):
        super().__init__()

        self.bert = model.from_pretrained(model_ckpt)
        self.pooler = AttentionPooling(self.bert.config.num_hidden_layers, self.bert.config.hidden_size, hiddendim_fc)
        self.drop = get_dropouts(num = CFG.num_drop, start_prob = CFG.hidden_dropout_prob, increment= CFG.increment_dropout_prob)
        self.classifier = nn.Linear(hiddendim_fc, CFG.num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
        all_hidden_states = torch.stack(bert_output[2])
        attention_pooling_embeddings = self.pooler(all_hidden_states)

        num_dps = float(len(self.drop))

        for ii, drop in enumerate(self.drop):
            if ii == 0:
                logits = (self.classifier(drop(attention_pooling_embeddings)) / num_dps)
            else :
                logits += (self.classifier(drop(attention_pooling_embeddings)) / num_dps)
        
        return logits

class WLPDropModel(nn.Module):
    def __init__(self, model, model_ckpt, layer_start = 9):
        super().__init__()

        self.bert = model.from_pretrained(model_ckpt)
        self.pooler = WeightedLayerPooling(num_hidden_layers = self.bert.config.num_hidden_layers, layer_start = layer_start, layer_weights=None)
        self.drop = get_dropouts(num = CFG.num_drop, start_prob = CFG.hidden_dropout_prob, increment= CFG.increment_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, CFG.num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
        all_hidden_states = torch.stack(bert_output[2])
        weighted_pooling_embeddings = self.pooler(all_hidden_states)[:, 0]

        num_dps = float(len(self.drop))

        for ii, drop in enumerate(self.drop):
            if ii == 0:
                logits = (self.classifier(drop(weighted_pooling_embeddings)) / num_dps)
            else :
                logits += (self.classifier(drop(weighted_pooling_embeddings)) / num_dps)
        
        return logits

    
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
        _, max_pool = torch.max(outputs.last_hidden_state, 1)
        x = torch.cat((avg_pool, max_pool), 1)
        x = self.dropout(x)
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
        self.classifier = nn.Linear(4 * self.model.config.hidden_size, CFG.num_labels)

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

class CatModel(nn.Module):    
    def __init__(self, model, model_ckpt):        
        super(CatModel, self).__init__()        
        self.bert = model.from_pretrained(model_ckpt)        
        self.dropout = nn.Dropout(0.5)        
        self.classifier = nn.Linear(4 * self.bert.config.hidden_size, CFG.num_labels)    

    def forward(self, input_ids, attention_mask):        
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)        
        cat_output = torch.cat((bert_output[2][-1][:,0, ...], bert_output[2][-2][:,0, ...], bert_output[2][-3][:,0, ...], bert_output[2][-4][:,0, ...]),-1)       
        outputs = self.dropout(cat_output)        
        outputs = self.classifier(outputs)
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
    elif model_type == "attention":
        return AttentionModel(model, model_ckpt)
    elif model_type == "weight_pool":
        return WLPDropModel(model, model_ckpt)
    elif model_type == "catmodel":
        return CatModel(model, model_ckpt)
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
