"""
Dataset
"""
from array import array

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from config import CFG
from tqdm import tqdm

# Sorry for this very very silly code, to much magic number, dummy
# maybe clean after...
# convert multi-label to one-hot
# Ex: [2,3,1,0] -> [0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0]

def convert_line(text, vocab, bpe, max_sequence_length):
    outputs = np.zeros((1, max_sequence_length))
    
    cls_id = 0
    eos_id = 2
    pad_id = 1

    subword = bpe.encode('<s> ' + text + ' </s>')
    input_ids = vocab.encode_line(subword, append_eos=False, add_if_not_exist=False).long().tolist()
    if len(input_ids) > max_sequence_length:
        input_ids = input_ids[:max_sequence_length] 
        input_ids[-1] = eos_id
    else:
        input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
    outputs = np.array(input_ids)
    return outputs

def convert_label(rating:int) -> np.array:
    """convert label of 1 aspect into one-hot vector
        Ex: 2 -> [0,1,0,0,0]
    Args:
        rating (int): rating

    Returns:
        np.array: one-hot vector
    """
    label = np.array([0, 0, 0, 0, 0])
    if rating:
        label[rating-1] = 1
    return label

def get_multi_label(labels:array) -> np.array:
    """convert multi-label to one-hot vector
    Ex: [2,3,1,0] -> [0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0]
    Args:
        labels (array): labels of multi-aspects get from the dataframe

    Returns:
        np.array: one-hot vector
    """
    multi_labels = np.array([])
    for label in labels:
        multi_labels = np.append(multi_labels, convert_label(label))
    return multi_labels

class HackathonDataset(Dataset):
    '''Custom Dataset
        Return:
        data['input_ids']: torch.Size([batch_size, max_len])
        data['attention_mask']: torch.Size([batch_size, max_len])
        targets: torch.Size([batch_size, num_labels])
    '''
    def __init__(self, df: DataFrame, tokenizer: Tokenizer, is_label=True, is_segmented=False) -> None:
        """init

        Args:
            df (DataFrame): dataframe
            tokenizer (Tokenizer): tokenizer, get from get_tokenizer
            is_train (bool, optional): is_train or not. Defaults to True.
        """
        super().__init__()
        self.df = df
        # reset index to avoid error when __getitem__
        self.df = self.df.reset_index()
        if is_segmented:
            self.texts = df["Review_segmented"].values
        else:
            self.texts = df["Review"].values
        self.tokenizer = tokenizer
        self.is_label = is_label
        self.aspects = ["giai_tri","luu_tru","nha_hang","an_uong","di_chuyen","mua_sam"]
        # multi_labels = [[self.df.loc[index, aspect] for aspect in self.aspects] for index in range(len(self.df))]
        self.labels = self.get_target()
        # self.vocab = vocab
        # self.bpe = bpe
            
    def get_target(self):

        df_dum = pd.get_dummies(self.df, columns = self.aspects)

        drop_col = []
        for col in df_dum.columns:
            if '0' in col or 'aspect' in col:
                drop_col.append(col)

        df_dum.drop(drop_col, axis = 1, inplace = True) 
        target_col = [f"{aspect}_{rating}" for aspect in self.aspects for rating in range(1, 6)]
        
        return df_dum[target_col].values

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        # encoding = {}
        # encoding['input_ids'] = convert_line(text, self.vocab, self.bpe, CFG.max_len)
        # encoding['attention_mask'] = encoding['input_ids'] > 0
        encoding = self.tokenizer(text, max_length=CFG.max_len,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        encoding['input_ids'] = torch.tensor(encoding['input_ids']).flatten()
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask']).flatten()
        if self.is_label:
            # Get multi-labels of multi-aspects from the dataframe
            # multi_labels = [self.df.loc[index, aspect] for aspect in self.aspects]
            labels = self.labels[index]
            return encoding, torch.tensor(labels, dtype=torch.float)
        return encoding
    
def create_dataloader(df: DataFrame, tokenizer: Tokenizer, batch_size: int, is_label=True, is_train=True, is_segmented=False) -> DataLoader:
    """create dataloader

    Args:
        df (DataFrame): input dataframe
        tokenizer (Tokenizer): tokenizer, get from get_tokenizer
        batch_size (int): batch size
        is_label (bool, optional): to avoid mistake, is the dataframe has label or not. Defaults to True.
        is_train (bool, optional): is the dataloader is train or not. Defaults to True.

    Returns:
        DataLoader: return dataloader
    """
    dataset = HackathonDataset(df, tokenizer, is_label, is_segmented)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True if is_train else False, num_workers=2)
