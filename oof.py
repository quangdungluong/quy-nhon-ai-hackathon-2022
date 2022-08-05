import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import argparse

def get_precision_recall_f1_score(labels:np.array, predictions:np.array) -> tuple:
    """Calculate precision, recall and f1 score

    Args:
        labels (np.array): grounth truth
        predictions (np.array): predictions

    Returns:
        tuple: (precision, recall, f1_score)
    """
    labels_ = labels > 0
    predictions_ = predictions > 0
    return (precision_score(labels_, predictions_), recall_score(labels_, predictions_), f1_score(labels_, predictions_))

def get_r2_score(labels:np.array, predictions:np.array) -> float:
    """Calculate r2 score

    Args:
        labels (np.array): grounth truth
        predictions (np.array): predictions

    Returns:
        float: r2_score
    """
    max_sentiment = 5
    min_sentiment = 1
    rss = 0
    n = 0
    for i in range(len(labels)):
        if labels[i] * predictions[i] > 0:
            rss += (labels[i] - predictions[i])**2
            n += 1
    try :
        return 1 - rss/(n * (max_sentiment-min_sentiment)**2)
    except :
        return 1 # competition rules

def report_score(df:pd.DataFrame, aspect:str) -> dict:
    """Report score for aspect

    Args:
        df (pd.DataFrame): oof_file
        aspect (str): aspect

    Returns:
        dict: score for this aspect
    """
    labels = df[aspect].values
    predictions = df[f"{aspect}_prediction"].values
    score = {}
    score["precision"], score["recall"], score["f1_score"] = get_precision_recall_f1_score(labels, predictions)
    score["r2_score"] = get_r2_score(labels, predictions)
    score["competition_score"] = score["f1_score"] * score["r2_score"]
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument Parser for Calculate CV score")
    parser.add_argument('--folder', type=str, default='03_08_2022', help="select subfolder of oof files")
    parser.add_argument('--save', type=bool, default=True, help="save report score or not")
    args = parser.parse_args()

    score_dict = {}
    aspects = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    root_dir = "./oof/"
    for aspect in aspects:
        oof_file = os.path.join(root_dir, args.folder, f"oof_{aspect}.csv")
        if os.path.exists(oof_file):
            oof_df = pd.read_csv(oof_file)
            score_dict[aspect] = report_score(oof_df, aspect)
    print(f"Score dict: {score_dict}\n")

    df = pd.DataFrame.from_dict(score_dict, orient="index")
    print(df)
    print()
    if args.save:
        df.to_csv(os.path.join(root_dir, args.folder, "oof_score.csv"))

    final_score = 0
    for score in score_dict.values():
        final_score += score["competition_score"] / len(score_dict)
    print(f"Final score: {final_score}\n")
