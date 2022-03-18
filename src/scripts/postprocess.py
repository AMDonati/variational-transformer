import argparse
import os
import pandas as pd
import json
import numpy as np
from datasets import load_metric
from src.eval.metrics import gpt2_perplexity_batch

bleu_metric = load_metric('google_bleu')
bertmetric = load_metric("bertscore")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument("-inference_prefix", type=str, default='test',
                        help="name of csv path for inference text samples")
    parser.add_argument("-to_remove", type=str, default="var_bleu")
    # parser.add_argument('-bottom_folder', type=int, default=1)
    # parser.add_argument('-top_folder', type=int, default=1)
    parser.add_argument('-precision', type=int, default=2)
    parser.add_argument('-test_scores', type=int, default=0)
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


def compute_test_metrics(texts):
    test_scores = dict.fromkeys(["bleu", "bertscore", "gpt2ppl"])
    for index in texts.index:
        bleu_metric.add_batch(predictions=[texts.loc[index]["preds"].split()],
                              references=[[texts.loc[index]["targets"].split()]])
    bleuscore = bleu_metric.compute()
    print("bleu score", bleuscore)
    test_scores["bleu"] = bleuscore["google_bleu"]
    predictions = list(texts["preds"])
    references = list(texts["targets"])
    results = bertmetric.compute(predictions=predictions, references=references, lang="en")
    bertscore = np.mean(results["f1"])
    print("bert score", bertscore)
    test_scores["bertscore"] = bertscore
    batch_size = 16
    gpt2ppl = []
    num_batches = np.floor(len(predictions) / batch_size).astype(np.int32)
    remains = len(predictions) - batch_size * num_batches
    for i in range(num_batches):
        # print("{}:{}".format(i*batch_size,(i+1)*batch_size))
        batch = predictions[i * batch_size:(i + 1) * batch_size]
        gpt2ppl_batch = gpt2_perplexity_batch(batch)
        gpt2ppl.append(gpt2ppl_batch)
    #gpt2ppl.append(gpt2_perplexity_batch(predictions[remains:]))
    gpt2ppl = np.mean(gpt2ppl)
    print("gpt2 ppl", gpt2ppl)
    test_scores["gpt2ppl"] = gpt2ppl
    return pd.DataFrame.from_records(test_scores, index=["scores"])

def compute_selfbleu(texts_multi):
    i = 0
    while 10 * i <= len(texts_multi.index):
        sentences = texts_multi.loc[i * 10:(i + 1) * 10 - 1]["preds"].to_list()
        for j, sentence in enumerate(sentences):
            ref_sentences = np.delete(sentences, j)
            ref_sentences = [s.split() for s in ref_sentences]
            bleu_metric.add_batch(predictions=[sentence.split()],
                                  references=[ref_sentences])
        i += 1
    bleuscore = bleu_metric.compute()
    return bleuscore["google_bleu"]


def merge_one_experiment(args):
        #path="output/temp", precision=4, to_remove="var_bleu", inference_prefix="test"):
    dirs = [f.path for f in os.scandir(args.path) if f.is_dir()]
    merge_metrics = pd.DataFrame()
    merge_scores = pd.DataFrame()
    texts = {}
    for i, dir_conf in enumerate(dirs):
        dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
        metrics_all_runs = []
        for dir_experiment in dirs:  # level for multiple runs with same config.
            config = open_config(os.path.join(dir_experiment, "config.json"))
            train_history_csv = os.path.join(dir_experiment, "train_history.csv")
            train_history = pd.read_csv(train_history_csv)
            exp_path = os.path.basename(dir_conf)
            val_loss_key = "val_loss" if "transformer" in exp_path else "val_ce_loss"
            best_val_loss = pd.Series(min(train_history[val_loss_key]), name="best_val_ce_loss")
            metrics = pd.DataFrame(train_history.values[-1, 1:][np.newaxis, :],
                                   columns=transform_columns(train_history.columns[1:],
                                                             path=exp_path))
            metrics = pd.concat([metrics, best_val_loss], axis=1)
            metrics_all_runs.append(metrics)
            text = pd.read_csv(os.path.join(dir_experiment, "inference", "texts_{}.csv".format(args.inference_prefix)))
            if args.test_scores:
                test_scores = compute_test_metrics(text)
                text_multi_path = os.path.join(dir_experiment, "inference", "texts_multi_{}_sampling.csv".format(args.inference_prefix))
                if os.path.exists(text_multi_path):
                    texts_multi = pd.read_csv(text_multi_path)
                    selfbleu = compute_selfbleu(texts_multi)
                    print("self bleu - sampling", selfbleu)
                    #test_scores = pd.DataFrame.from_records({"selfbleu_sampling": selfbleu}, index=["scores"])
                    test_scores["selfbleu_sampling"] = selfbleu
                text_multi_path = os.path.join(dir_experiment, "inference",
                                               "texts_multi_{}_greedy.csv".format(args.inference_prefix))
                if os.path.exists(text_multi_path):
                    texts_multi = pd.read_csv(text_multi_path)
                    selfbleu = compute_selfbleu(texts_multi)
                    print("self bleu - greedy", selfbleu)
                    test_scores["selfbleu_greedy"] = selfbleu
        if len(dirs) > 1:
            metrics_all_runs = pd.concat(metrics_all_runs)
            metrics = metrics_all_runs.mean(axis=0).apply(lambda t: round(t, args.precision), axis=1)
            std_metrics = metrics_all_runs.std(axis=0).apply(lambda t: round(t, 3), axis=1)
        else:
            metrics = metrics_all_runs[-1].apply(lambda t: round(t, args.precision), axis=1)
        metrics.to_csv(os.path.join(dir_conf, "all_metrics.csv"))
        if args.test_scores:
            test_scores.to_csv(os.path.join(dir_conf, "test_scores.csv"))
        merge_metrics[os.path.basename(dir_conf)] = metrics.T.squeeze()
        texts[os.path.basename(dir_conf)] = text
        if args.test_scores:
            merge_scores[os.path.basename(dir_conf)] = test_scores.T.squeeze()
    merge_texts_1 = texts[list(texts.keys())[0]][["inputs", "targets"]]
    texts_ = {k: v["preds"] for k, v in texts.items()}
    merge_texts_2 = pd.DataFrame.from_records(texts_)
    merge_texts = pd.concat([merge_texts_1, merge_texts_2], axis=1)
    merge_metrics.to_csv(os.path.join(args.path, "merge_metrics.csv"))
    if args.test_scores:
        merge_scores.to_csv(os.path.join(args.path, "merge_scores.csv"))
    merge_texts.to_csv(os.path.join(args.path, "merge_texts.csv"))
    # if to_remove in merge_metrics.index:
    # merge_metrics = merge_metrics.drop(to_remove, axis=0)
    # merge_metrics_latex = merge_metrics.apply(lambda t: t.replace('+/-', '\pm'))
    merge_metrics_latex = merge_metrics
    merge_metrics_latex.columns = [col.replace('_', '-') for col in merge_metrics_latex.columns]
    merge_metrics_latex.index = [ind.replace('_', '-') for ind in merge_metrics_latex.index]
    merge_metrics_latex = merge_metrics_latex.T
    merge_metrics_latex.to_latex(os.path.join(args.path, "merge_metrics.txt"))



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    merge_one_experiment(args)

    # path = "output/temp/VAE_1L_d32_dff32_pe1000_bs32_pdrop0.1_attention_None10-1.0/20220228-171311/inference/texts_multi_test.csv"
    # texts_multi = pd.read_csv(path)
    #
    # bleu_metric = load_metric('google_bleu')
    # i = 0
    # while 10*i <= len(texts_multi.index):
    #     sentences = texts_multi.loc[i*10:(i+1)*10-1]["preds"].to_list()
    #     for j, sentence in enumerate(sentences):
    #         ref_sentences = np.delete(sentences, j)
    #         ref_sentences = [s.split() for s in ref_sentences]
    #         bleu_metric.add_batch(predictions=[sentence.split()],
    #                               references=[ref_sentences])
    #     i += 1
    # bleuscore = bleu_metric.compute()
    # print("done")
