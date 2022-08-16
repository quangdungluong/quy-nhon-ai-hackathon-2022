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

class HackathonDataset(Dataset):
    '''Custom Dataset
        Return:
        data['input_ids']: torch.Size([batch_size, max_len])
        data['attention_mask']: torch.Size([batch_size, max_len])
        targets: torch.Size([batch_size, num_labels])
    '''
    def __init__(self, df: DataFrame, tokenizer: Tokenizer, is_label=True, is_segmented=True) -> None:
        """init

        Args:
            df (DataFrame): dataframe
            tokenizer (Tokenizer): tokenizer, get from get_tokenizer
            is_label (bool, optional): is train or not. Defaults to True.
            is_segmented (bool, optional): is using word segmented or not. Defaults to True.
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
        self.labels = self.get_target()

        # preprocessing
        self.vocab = self.tokenizer.get_vocab()

        words_out = pd.read_csv('./out_word.csv')
        origin_words = words_out['out_word'].values
        replace_words = words_out['replace'].values
        check_list = {}
        for i in range(len(origin_words)):
            check_list[origin_words[i]] = replace_words[i]
        self.check_list = check_list
            
    def get_target(self):
        df_dum = pd.get_dummies(self.df, columns = self.aspects)
        drop_col = []
        for col in df_dum.columns:
            if '0' in col or 'aspect' in col:
                drop_col.append(col)
        df_dum.drop(drop_col, axis = 1, inplace = True) 
        target_col = [f"{aspect}_{rating}" for aspect in self.aspects for rating in range(1, 6)]
        
        labels = df_dum[target_col].values
        labels = self.label_smoothing(labels)
        return labels

    def label_smoothing(self, labels):
        smoothing = [0.5, 0.1, 0.02, 0.01]
        LS = []
        for label in labels:
            label_ = np.ones((30,))
            for i in range(6): #loop aspect
                index = -1
                for j in range(5): #loop rating
                    if (label[5*i+j] == 1):
                        index = j
                for j in range(1,5):
                    if index - j >= 0:
                        label_[5*i + index - j] = smoothing[j-1]
                    if index + j < 5:
                        label_[5*i + index + j] = smoothing[j-1]
            LS.append(label_)
        return LS

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        lower_text = text.lower()
        list_preprocessed_words = []
        list_word = lower_text.split()
        for word in list_word:
            if word in self.vocab or word not in self.check_list:
                list_preprocessed_words.append(str(word))
            else:
                list_preprocessed_words.append(str(self.check_list[word]))
        text = ' '.join(list_preprocessed_words)
        
        encoding = self.tokenizer(text, max_length=CFG.max_len,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        encoding['input_ids'] = torch.tensor(encoding['input_ids']).flatten()
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask']).flatten()
        if self.is_label:
            # Get multi-labels of multi-aspects from the dataframe
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
        is_segmented (bool, optional): is using word segmented or not. Defaults to False.

    Returns:
        DataLoader: dataloader
    """
    dataset = HackathonDataset(df, tokenizer, is_label, is_segmented)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True if is_train else False, num_workers=2)
