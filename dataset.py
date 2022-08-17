"""
Dataset
"""

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
    def __init__(self, df: DataFrame, tokenizer: Tokenizer, is_label=True, is_segmented=True, is_preprocessing=True) -> None:
        """init

        Args:
            df (DataFrame): dataframe
            tokenizer (Tokenizer): tokenizer, get from get_tokenizer
            is_label (bool, optional): is train or not. Defaults to True.
            is_segmented (bool, optional): is using word segmented or not. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.df = self.df.reset_index() # reset index to avoid error when __getitem__
        self.tokenizer = tokenizer
        self.is_label = is_label
        self.is_preprocessing = is_preprocessing
        self.aspects = ["giai_tri","luu_tru","nha_hang","an_uong","di_chuyen","mua_sam"]
        self.labels = self.get_target()
        if is_segmented:
            self.texts = df["Review_segmented"].values
        else:
            self.texts = df["Review"].values

        # preprocessing
        words_out = pd.read_csv('./out_word.csv')
        origin_words = words_out['out_word'].values
        replace_words = words_out['replace'].values
        check_list = {}
        for i in range(len(origin_words)):
            check_list[origin_words[i]] = replace_words[i]
        self.check_list = check_list
        self.vocab = self.tokenizer.get_vocab()
            
    def get_target(self):
        df_dum = pd.get_dummies(self.df, columns = self.aspects)
        drop_col = []
        for col in df_dum.columns:
            if '0' in col or 'aspect' in col:
                drop_col.append(col)
        df_dum.drop(drop_col, axis = 1, inplace = True) 
        target_col = [f"{aspect}_{rating}" for aspect in self.aspects for rating in range(1, 6)]
        labels = df_dum[target_col].values
        return labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        if self.is_preprocessing:
            lower_text = text.lower()
            list_word = lower_text.split()
            list_preprocessed_words = []
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
        if self.is_label:
            labels = self.labels[index] # Get multi-labels of multi-aspects from the dataframe
            return {
                'input_ids': torch.tensor(encoding['input_ids']).flatten(),
                'attention_mask': torch.tensor(encoding['attention_mask']).flatten(),
                'target': torch.tensor(labels, dtype=torch.float)
            }
        return {
            'input_ids': torch.tensor(encoding['input_ids']).flatten(),
            'attention_mask': torch.tensor(encoding['attention_mask']).flatten()
        }
    
def create_dataloader(df: DataFrame, tokenizer: Tokenizer, batch_size: int, is_label=True, is_train=True, is_segmented=False, is_preprocessing=True) -> DataLoader:
    """create dataloader

    Args:
        df (DataFrame): input dataframe
        tokenizer (Tokenizer): tokenizer, get from get_tokenizer
        batch_size (int): batch size
        is_label (bool, optional): to avoid mistake, is the dataframe has label or not. Defaults to True.
        is_train (bool, optional): is the dataloader is train or not. Defaults to True.
        is_segmented (bool, optional): is using word segmented or not. Defaults to False.
        is_preprocessing (bool, optional): is using preprocessing or not. Defaults to True.

    Returns:
        DataLoader: dataloader
    """
    dataset = HackathonDataset(df, tokenizer, is_label, is_segmented, is_preprocessing)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True if is_train else False, num_workers=2)
