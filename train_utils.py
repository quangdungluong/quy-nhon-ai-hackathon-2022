"""
Training Utility Functions
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import create_dataloader
from metrics import get_score
from models import create_model, create_tokenizer
from utils import (get_final_prediction, get_group_optimizer,
                   get_layerwise_lr_decay, get_optimizer, get_scheduler)


def train_epoch(model:nn.Module, dataloader:DataLoader, optimizer:Optimizer, scheduler):
    """Training 1 epoch

    Args:
        model (nn.Module): model
        dataloader (DataLoader): dataloader
        optimizer (Optimizer): optimizer
        scheduler (Scheduler): scheduler

    Returns:
        _type_: loss, score dictionary, final competition score
    """
    model.train()
    losses = []
    labels = None
    predictions = None
    
    for data in tqdm(dataloader):
        input_ids = data["input_ids"].to(CFG.device)
        attention_mask = data["attention_mask"].to(CFG.device)
        targets = data['target'].to(CFG.device)
        targets_smoothing = data['target_smoothing'].to(CFG.device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = F.binary_cross_entropy_with_logits(outputs, targets_smoothing.float())
        loss = loss.mean()
        loss.backward() 
        losses.append(loss.item())

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        targets = targets.squeeze().detach().cpu().numpy()
        labels = np.atleast_2d(targets) if labels is None else np.concatenate([labels, np.atleast_2d(targets)])
        outputs = outputs.squeeze().detach().cpu().numpy()
        predictions = np.atleast_2d(outputs) if predictions is None else np.concatenate([predictions, np.atleast_2d(outputs)])
        
    loss = np.mean(losses)
    score_dict = get_score(labels, predictions)
    final_score = 0
    for score in score_dict.values():
        final_score += score["competition_score"] / len(score_dict)
    return loss, score_dict, final_score

def eval_model(model:nn.Module, dataloader:DataLoader):
    """Evaluation after 1 training epoch

    Args:
        model (nn.Module): model
        dataloader (DataLoader): dataloader

    Returns:
        _type_: loss, score dictionary, final competition score
    """
    model.eval()
    losses = []
    labels = None
    predictions = None
    
    for data in tqdm(dataloader):
        input_ids = data["input_ids"].to(CFG.device)
        attention_mask = data["attention_mask"].to(CFG.device)
        targets = data['target'].to(CFG.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
        loss = loss.mean()
        losses.append(loss.item())
        
        targets = targets.squeeze().detach().cpu().numpy()
        labels = np.atleast_2d(targets) if labels is None else np.concatenate([labels, np.atleast_2d(targets)])
        outputs = outputs.squeeze().detach().cpu().numpy()
        predictions = np.atleast_2d(outputs) if predictions is None else np.concatenate([predictions, np.atleast_2d(outputs)])
        
    loss = np.mean(losses)
    score_dict = get_score(labels, predictions)
    final_score = 0
    for score in score_dict.values():
        final_score += score["competition_score"] / len(score_dict)
    return loss, score_dict, final_score

def train_fold(model_name:str, model_type:str, scheduler_type:str, fold:int, train_fold_df:pd.DataFrame, val_fold_df:pd.DataFrame, oof_file:pd.DataFrame, model_ckpt:str, is_segmented=False, save_last=False) -> pd.DataFrame:
    """Training 1 fold

    Args:
        model_name (str): model_name
        model_type (str): model_type
        scheduler_type (str): scheduler type
        aspect (str): aspect for training
        fold (int): fold to train
        train_fold_df (pd.DataFrame): train fold dataframe
        val_fold_df (pd.DataFrame): val fold dataframe
        oof_file (pd.DataFrame): out of fold dataframe
        model_ckpt (str): path to model-checkpoint directory
        is_segmented (bool, optional): is using word segmented or not. Defaults to False.
        save_last (bool, optional): save last epoch or not. Defaults to False.

    Returns:
        pd.DataFrame: oof_file
    """
    print('='*10 + f'Fold {fold}' + '='*10)
    print()

    # Create model and tokenizer
    model = create_model(model_name, model_type)
    model = model.to(CFG.device)
    tokenizer = create_tokenizer(model_name)
    
    # Create dataloader
    train_dataloader = create_dataloader(train_fold_df, tokenizer, CFG.batch_size, is_train=True, is_segmented=is_segmented)
    val_dataloader = create_dataloader(val_fold_df, tokenizer, CFG.batch_size, is_train=False, is_segmented=is_segmented)
    
    # Create optimizer, scheduler, loss function
    if CFG.is_llrd : 
        grouped_optimizer_params  = get_layerwise_lr_decay(model)
        optimizer = optim.AdamW(grouped_optimizer_params, lr = CFG.lr, correct_bias = True)
    else :
        if CFG.optimizer_type == "basic":
            optimizer = get_optimizer(model)
        else:
            optimizer = get_group_optimizer(model)

    scheduler = get_scheduler(optimizer, scheduler_type, len(train_dataloader)*CFG.num_epochs)
    # Init to save best model
    best_score = -999
    best_epoch = 0
    
    # Training
    for epoch in range(CFG.num_epochs):
        print('-'*5 + f'Epoch {epoch+1}/{CFG.num_epochs}' + '-'*5)
        # Train phase
        train_loss, score, competition_score = train_epoch(model, train_dataloader, optimizer, scheduler)
        score_report = pd.DataFrame.from_dict(score, orient="index")
        print(f"Train loss: {train_loss:.4f} competition_score: {competition_score:.4f}")
        print(score_report)
        print()
        # Eval phase
        val_loss, score, competition_score = eval_model(model, val_dataloader)
        score_report = pd.DataFrame.from_dict(score, orient="index")
        print(f"Valid loss: {val_loss:.4f} competition_score: {competition_score:.4f}")
        print(score_report)
        print()
        # Save best
        if competition_score > best_score:
            print(f"Improved!!! {best_score:.4f} --> {competition_score:.4f}\n")
            best_score = competition_score
            best_epoch = epoch + 1
            save_path = os.path.join(model_ckpt, f'best_model_fold_{fold}.bin')
            torch.save(model.state_dict(), save_path)
    
    print(f'Training Completed!\nBest val competition score: {best_score:.4f} at epoch {best_epoch}.')

    # Save last checkpoint
    if save_last:
        save_path = os.path.join(model_ckpt, f'last_model_fold_{fold}.bin')
        torch.save(model.state_dict(), save_path)
        print('Saved last model checkpoint!') 

    # Save OOF file
    model.load_state_dict(torch.load(os.path.join(model_ckpt, f'best_model_fold_{fold}.bin'))) # load best model
    model = model.eval()
    logits = [] # a list has len equal num examples and logits[0] has shape (30,)

    for data in tqdm(val_dataloader):
        input_ids = data['input_ids'].to(CFG.device)
        attention_mask = data['attention_mask'].to(CFG.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        for out in outputs.detach().cpu().numpy():
            logits.append(out)
                 
    # logits is a list has num_examples len and each is a np.array of size (30, )
    predictions, probs = get_final_prediction(logits)
    
    oof_file.loc[val_fold_df.index, ["giai_tri_prediction","luu_tru_prediction","nha_hang_prediction","an_uong_prediction","di_chuyen_prediction","mua_sam_prediction"]] = predictions
    aspects = ["giai_tri","luu_tru","nha_hang","an_uong","di_chuyen","mua_sam"]
    for aspect in aspects:
        oof_file.loc[val_fold_df.index, [f'{aspect}_1', f'{aspect}_2', f'{aspect}_3', f'{aspect}_4', f'{aspect}_5']] = probs[aspect]

    torch.cuda.empty_cache()

    return oof_file
