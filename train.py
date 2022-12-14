import argparse
import os
import warnings

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from config import CFG
from train_utils import train_fold
from utils import preprocess, seed_everything

warnings.filterwarnings("ignore")
import transformers

transformers.logging.set_verbosity_error()
from vncorenlp import VnCoreNLP


def main():
    train_df = pd.read_csv(CFG.train_path)
    is_segmented = False

    if CFG.preprocess:
        train_df["Review"] = train_df["Review"].apply(lambda x: preprocess(x))
        
    if CFG.rdrsegmenter_path is not None:
        is_segmented = True
        rdrsegmenter = VnCoreNLP(CFG.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 
        train_df["Review_segmented"] = train_df["Review"].apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
    
    # Split data using MultilabelStratifiedKFold
    aspects = ["giai_tri","luu_tru","nha_hang","an_uong","di_chuyen","mua_sam"]
    kfold = MultilabelStratifiedKFold(n_splits=CFG.num_folds,
                            shuffle=True, random_state=CFG.seed)
    folds = train_df.copy()
    targets = folds[aspects].values
    for fold, (train_index, val_index) in enumerate(kfold.split(folds, targets)):
        folds.loc[val_index, 'fold'] = fold
    folds['fold'] = folds['fold'].astype(int)

    oof_file = train_df.copy()

    for fold in CFG.train_folds:
        train_fold_df = folds[folds['fold'] != fold]
        val_fold_df = folds[folds['fold'] == fold]
        oof_file = train_fold(model_name=CFG.model_name, model_type=CFG.model_type, scheduler_type=CFG.scheduler_type, 
                            fold=fold, train_fold_df=train_fold_df, val_fold_df=val_fold_df, 
                            oof_file=oof_file, model_ckpt=CFG.model_ckpt, is_segmented=is_segmented)

    # save oof_file
    oof_file.to_csv(os.path.join(CFG.output_path, f"oof.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument Parser for Training model")
    parser.add_argument('--model_name', type=str, default='phobert-base', help="select model name")
    parser.add_argument('--model_type', type=str, default="4_hidden", help="select model type")
    parser.add_argument('--train_path', type=str,
                        default='./train_final.csv', help='training df path')
    parser.add_argument('--model_ckpt', type=str,
                        default="/kaggle/working/ckpt", help="path to model ckpt folder")
    parser.add_argument("--output_path", type=str,
                        default="/kaggle/working/output", help="path to output folder")
    parser.add_argument("--rdrsegmenter_path", type=str, default=None, help="rdrsegmenter path")
    parser.add_argument("--num_epochs", type=int, default=15, help="change number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="change learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="change hidden dropout probability")
    parser.add_argument("--increment_dropout_prob", type=float, default=0.1, help="increment_dropout_prob")
    parser.add_argument("--train_folds", nargs='+', type=int, default=[0, 1, 2, 3, 4], help="choose train folds")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="choose scheduler types")
    parser.add_argument("--batch_size", type=int, default=4, help="choose batch size")
    parser.add_argument("--seed", type=int, default=42, help="choose seed")
    parser.add_argument("--is_smoothing", type=bool, default=True, help="is using label smoothing or not")
    parser.add_argument("--num_folds", type=int, default=5, help="number of folds")
    parser.add_argument("--optimizer_type", type=str, default="basic", help="choose optimizer type, group or basic")
    parser.add_argument("--preprocess", type=bool, default=False, help="replace word or not")
    parser.add_argument("--smoothing", nargs='+', type=float, default=[0.6, 0.2, 0.1, 0.05], help="choose smoothing params")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="number of warmup steps in scheduler")
    args = parser.parse_args()
    
    
    if "phobert" in args.model_name:
        CFG.max_len = 256
    CFG.train_folds = args.train_folds
    CFG.model_name = args.model_name
    CFG.model_type = args.model_type
    CFG.train_path = args.train_path
    CFG.model_ckpt = args.model_ckpt
    CFG.output_path = args.output_path
    CFG.rdrsegmenter_path = args.rdrsegmenter_path
    CFG.lr = args.lr
    CFG.hidden_dropout_prob = args.dropout
    CFG.num_epochs = args.num_epochs
    CFG.scheduler_type = args.scheduler_type
    CFG.increment_dropout_prob = args.increment_dropout_prob
    CFG.batch_size = args.batch_size
    CFG.seed = args.seed
    CFG.is_smoothing = args.is_smoothing
    CFG.num_folds = args.num_folds
    CFG.optimizer_type = args.optimizer_type
    CFG.preprocess = args.preprocess
    CFG.smoothing = args.smoothing

    print(f'Seed {CFG.seed}')
    seed_everything(CFG.seed)
    
    if not os.path.exists(args.model_ckpt):
        os.makedirs(args.model_ckpt)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main()

    
