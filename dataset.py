from pandas import DataFrame
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset

from config import CFG
import numpy as np

def convert_label(rating:int) -> np.array:
    """convert label of 1 aspect into one-hot

    Args:
        rating (int): rating

    Returns:
        np.array: one-hot
    """
    label = np.array([0, 0, 0, 0, 0])
    if rating:
        label[rating-1] = 1
    return label

def get_multi_label(labels):
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
    def __init__(self, df: DataFrame, tokenizer: Tokenizer, is_label=True) -> None:
        """init

        Args:
            df (DataFrame): dataframe
            tokenizer (Tokenizer): tokenizer, get from get_tokenizer
            is_train (bool, optional): is_train or not. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.df = self.df.reset_index()
        self.texts = df["Review"].values
        self.tokenizer = tokenizer
        self.is_label = is_label
        self.aspects = ["giai_tri","luu_tru","nha_hang","an_uong","di_chuyen","mua_sam"]
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        encoding = self.tokenizer(text, max_length=CFG.max_len,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        encoding['input_ids'] = torch.tensor(encoding['input_ids']).flatten()
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask']).flatten()
        if self.is_label:
            multi_labels = [self.df.loc[index, aspect] for aspect in self.aspects]
            labels = get_multi_label(multi_labels)
            return encoding, torch.tensor(labels, dtype=torch.float)
        return encoding
    
def create_dataloader(df: DataFrame, tokenizer: Tokenizer, batch_size: int, is_label=True, is_train=True) -> DataLoader:
    """create dataloader

    Args:
        df (DataFrame): dataframe
        tokenizer (Tokenizer): tokenizer, get from get_tokenizer
        batch_size (int): batch size
        is_train (bool, optional): is train or not. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    dataset = HackathonDataset(df, tokenizer, is_label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True if is_train else False, num_workers=2)