import argparse
import os
import pandas as pd
import numpy as np
from metrics import report_score

def convert_probs_to_label(prediction_probs:np.array, threshold=0.5)->np.array:
    """convert prediction probs to predictions"""
    result = []
    for prediction_prob in prediction_probs:
        sub_result = np.array([0, 0, 0, 0, 0, 0])
        for i in range(6):
            best_score = -999
            index = -1
            for j in range(5):
                if prediction_prob[5*i+j] > best_score:
                    best_score = prediction_prob[5*i+j]
                    index = j
            if best_score >= threshold:
                sub_result[i] = index + 1
        result.append(sub_result)
    result = np.array(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument Parser for Calculate CV score")
    parser.add_argument('--folder', type=str, default='06-08-2022', help="select subfolder of oof files")
    parser.add_argument('--save', type=bool, default=True, help="save report score or not")
    args = parser.parse_args()

    aspects = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    columns = [f"{aspect}_{rating}" for aspect in aspects for rating in range(1,6)]
    oof_df = pd.read_csv(os.path.join("./oof/", args.folder, "oof.csv"))

    prediction_probs = np.array(oof_df[columns])
    labels = np.array(oof_df[aspects])

    score_dict = {}
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        sub_score = {}
        predictions = convert_probs_to_label(prediction_probs, threshold)
        print(predictions.shape)
        for i in range(6):
            sub_score[aspects[i]] = report_score(predictions[:,i], labels[:,i])
        final_score = 0
        for score in sub_score.values():
            final_score += score["competition_score"] / len(sub_score)
        print(f"Final score at threshold={threshold}: {final_score}\n")
        score_dict[threshold] = final_score
        df = pd.DataFrame.from_dict(sub_score, orient="index")
        print(df)
        print()
        if args.save:
            df.to_csv(os.path.join("./oof", args.folder, f"oof_score_{threshold}.csv"))
    if args.save:
        df = pd.DataFrame.from_dict(score_dict, orient="index")
        df.to_csv(os.path.join("./oof", args.folder, f"oof_score.csv"))
    print(score_dict)