import argparse
import os
import warnings

import pandas as pd
from sklearn.model_selection import KFold

from config import CFG
from dataset import create_dataloader
from models import create_model, create_tokenizer
from utils import seed_everything

warnings.filterwarnings("ignore")
import transformers

transformers.logging.set_verbosity_error()
import torch
from tqdm import tqdm

from utils import get_final_prediction


def main(args):
    train_df = pd.read_csv(args.train_path)

    # Split kfold
    kfold = KFold(n_splits=CFG.num_folds,
                            shuffle=True, random_state=CFG.seed)
    folds = train_df.copy()
    for fold, (train_index, val_index) in enumerate(kfold.split(folds)):
        folds.loc[val_index, 'fold'] = fold
    folds['fold'] = folds['fold'].astype(int)

    oof_file = train_df.copy()

    for fold in CFG.train_folds:
        val_fold_df = folds[folds['fold'] == fold]
        tokenizer = create_tokenizer(args.model_name)
        val_dataloader = create_dataloader(val_fold_df, tokenizer, CFG.batch_size, is_train=False)
        
        path = os.path.join(args.model_ckpt, f"best_model_fold_{fold}.bin")
        model = create_model(args.model_name, args.model_type)
        model = model.to(CFG.device)
        model.load_state_dict(torch.load(path))
        model = model.eval()
        logits = []
        
        for data, targets in tqdm(val_dataloader):
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
    oof_file.to_csv(os.path.join(args.output_path, f"oof.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument Parser for Training model")
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', help="select model name")
    parser.add_argument('--model_type', type=str, default="naive", help="select model type")
    parser.add_argument('--train_path', type=str,
                        default='./train_final.csv', help='training df path')
    parser.add_argument('--test_path', type=str,
                        default='./public_final.csv', help='test df path')
    parser.add_argument('--infer', type=bool, default=True,
                        help="infer after trained model or not, default=True")
    parser.add_argument('--model_ckpt', type=str,
                        default="/kaggle/working/ckpt", help="path to model ckpt folder")
    parser.add_argument("--submission", type=str,
                        default="submission.csv", help="path to submission file")
    parser.add_argument("--output_path", type=str,
                        default="/kaggle/working/output", help="path to output folder")
    parser.add_argument("--num_epochs", type=int, default=None, help="change number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="change learning rate")
    parser.add_argument("--dropout", type=float, default=None, help="change hidden dropout probability")
    args = parser.parse_args()

    print(f'Seed {CFG.seed}')
    seed_everything(CFG.seed)
    
    if "phobert" in args.model_name:
        CFG.max_len = 256
    if args.num_epochs is not None:
        CFG.num_epochs = args.num_epochs
    if args.lr is not None:
        CFG.lr = args.lr
    if args.dropout is not None:
        CFG.hidden_dropout_prob = args.dropout

    # if not os.path.exists(args.model_ckpt):
    #     os.makedirs(args.model_ckpt)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
