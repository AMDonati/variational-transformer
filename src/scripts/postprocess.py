import argparse
import os
import pandas as pd
import json
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument("-to_remove", type=str, default="var_bleu")
    # parser.add_argument('-bottom_folder', type=int, default=1)
    # parser.add_argument('-top_folder', type=int, default=1)
    parser.add_argument('-precision', type=int, default=2)
    return parser


def open_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def transform_columns(columns, path):
    new_columns = []
    if "transformer" in path:
        col1, col2 = "train_loss", "val_loss"
    elif "VAE" in path:
        col1, col2 = "train_ce_loss", "val_ce_loss"
    for col in columns:
        if col == col1:
            new_columns.append("train_ce")
        elif col == col2:
            new_columns.append("val_ce")
        else:
            new_columns.append(col)
    return new_columns


def add_missing_columns(df, path):
    if "transformer" in path:
        missing_cols = pd.DataFrame([["NA", "NA", "NA", "NA"]],
                                    columns=["train_loss", "val_loss", "train_kl_loss", "val_kl_loss"])
        new_df = pd.concat([df, missing_cols], axis=1)
    else:
        new_df = df
    return new_df


def merge_one_experiment(path="output/temp", precision=4, to_remove="var_bleu"):
    dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    merge_metrics = pd.DataFrame()
    texts = {}
    for i, dir_conf in enumerate(dirs):
        dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
        metrics_all_runs = []
        for dir_experiment in dirs:  # level for multiple runs with same config.
            config = open_config(os.path.join(dir_experiment, "config.json"))
            train_history_csv = os.path.join(dir_experiment, "train_history.csv")
            train_history = pd.read_csv(train_history_csv)
            metrics = pd.DataFrame(train_history.values[-1, 1:][np.newaxis, :],
                                   columns=transform_columns(train_history.columns[1:],
                                                             path=os.path.basename(dir_conf)))
            metrics_all_runs.append(metrics)
            text = pd.read_csv(os.path.join(dir_experiment, "inference", "texts.csv"))
        if len(dirs) > 1:
            metrics_all_runs = pd.concat(metrics_all_runs)
            metrics = metrics_all_runs.mean(axis=0).apply(lambda t: round(t, precision), axis=1)
            std_metrics = metrics_all_runs.std(axis=0).apply(lambda t: round(t, 3), axis=1)
        else:
            metrics = metrics_all_runs[-1].apply(lambda t: round(t, precision), axis=1)
        metrics.to_csv(os.path.join(dir_conf, "all_metrics.csv"))
        merge_metrics[os.path.basename(dir_conf)] = metrics.T.squeeze()
        texts[os.path.basename(dir_conf)] = text
    merge_texts_1 = texts[list(texts.keys())[0]][["inputs", "targets"]]
    texts_ = {k:v["preds"] for k, v in texts.items()}
    merge_texts_2 = pd.DataFrame.from_records(texts_)
    merge_texts = pd.concat([merge_texts_1, merge_texts_2], axis=1)
    merge_metrics.to_csv(os.path.join(path, "merge_metrics.csv"))
    merge_texts.to_csv(os.path.join(path, "merge_texts.csv"))
    #if to_remove in merge_metrics.index:
        #merge_metrics = merge_metrics.drop(to_remove, axis=0)
    # merge_metrics_latex = merge_metrics.apply(lambda t: t.replace('+/-', '\pm'))
    merge_metrics_latex = merge_metrics
    merge_metrics_latex.columns = [col.replace('_', '-') for col in merge_metrics_latex.columns]
    merge_metrics_latex.index = [ind.replace('_', '-') for ind in merge_metrics_latex.index]
    merge_metrics_latex = merge_metrics_latex.T
    merge_metrics_latex.to_latex(os.path.join(path, "merge_metrics.txt"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    merge_one_experiment(path=args.path, precision=args.precision, to_remove=args.to_remove)
