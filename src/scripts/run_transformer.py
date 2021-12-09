import argparse
from src.data_provider.ROCDataset import ROCDataset
from src.models.transformer import Transformer
from src.train.train_transformer import train
from src.train.utils import CustomSchedule, get_checkpoints
from src.eval.eval import inference
import tensorflow as tf
import os
import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import h5py


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    # parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-max_samples", type=int, help="max samples for train dataset")
    # model parameters:
    parser.add_argument("-num_layers", type=int, default=1,
                        help="number of layers in the network. If == 0, corresponds to adding GPT2Decoder.")
    parser.add_argument("-num_heads", type=int, default=1, help="number of attention heads for Transformer networks")
    parser.add_argument("-d_model", type=int, default=8, help="depth of attention parameters")
    parser.add_argument("-dff", type=int, default=8, help="dimension of feed-forward network")
    parser.add_argument("-pe", type=int, default=1000, help="maximum positional encoding")
    parser.add_argument("-p_drop", type=float, default=0.1, help="dropout on output layer")
    # training params.
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=5, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    # output_path params.
    parser.add_argument("-output_path", type=str, default="output", help="path for output folder")
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if loading ckpt)")
    # inference params.
    parser.add_argument("-test_samples", type=int, help="number of test samples.")
    parser.add_argument("-temp", type=float, default=0.7, help="temperature for sampling text.")

    return parser


def create_logger(out_path):
    out_file_log = os.path.join(out_path, 'training_log.log')
    logging.basicConfig(filename=out_file_log, level=logging.INFO)
    # create logger
    logger = logging.getLogger('training log')
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

def create_tensorboard_writer(out_path):
    train_log_dir = os.path.join(out_path, 'logs', 'train')
    val_log_dir = os.path.join(out_path, 'logs', 'val')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    return train_summary_writer, val_summary_writer


def save_hparams(args, out_path):
    dict_hparams = vars(args)
    dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
    config_path = os.path.join(out_path, "config.json")
    with open(config_path, 'w') as fp:
        json.dump(dict_hparams, fp, sort_keys=True, indent=4)


def create_ckpt_path(args, out_path):
    if args.save_path is not None:
        checkpoint_path = os.path.join(args.save_path, "checkpoints")
    else:
        checkpoint_path = os.path.join(out_path, "checkpoints")
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    return checkpoint_path


def create_out_path(args):
    if args.save_path is not None:
        return args.save_path
    else:
        out_file = 'Transformer_{}L_d{}_dff{}_pe{}_bs{}_pdrop{}'.format(args.num_layers, args.d_model, args.dff,
                                                                        args.pe,
                                                                        args.bs,
                                                                        args.p_drop)
        datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_folder = os.path.join(args.output_path, out_file, datetime_folder)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder


def plot_results(results, out_path):
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    x = np.linspace(1, len(train_loss), len(train_loss))
    plt.plot(x, train_loss, 'red', lw=2, label='train loss')
    plt.plot(x, val_loss, 'cyan', lw=2, label='val loss')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(out_path, "loss_plot.png"))


def run(args):
    # Create out path & logger & config.json file
    out_path = create_out_path(args)
    logger = create_logger(out_path)
    save_hparams(args, out_path)

    # Load Dataset
    dataset = ROCDataset(data_path='data/ROC', batch_size=args.bs, max_samples=args.max_samples)
    train_data, val_data, test_data = dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = dataset.data_to_dataset(train_data, val_data, test_data)

    vocab_size = len(dataset.vocab)

    # Create Transformer
    transformer = Transformer(
        num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads, dff=args.dff,
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        pe_input=args.pe, pe_target=args.pe)

    # Train Transformer
    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    checkpoint_path = create_ckpt_path(args, out_path)
    ckpt_manager = get_checkpoints(transformer, optimizer, checkpoint_path)
    results = train(EPOCHS=args.ep, train_dataset=train_dataset, val_dataset=val_dataset, transformer=transformer,
                    optimizer=optimizer, loss_object=loss_object, ckpt_manager=ckpt_manager, logger=logger)
    results["train_ppl"] = np.exp(results["train_loss"])
    results["val_ppl"] = np.exp(results["val_loss"])
    # save results
    df_results = pd.DataFrame.from_records(results)
    df_results.to_csv(os.path.join(out_path, "train_history.csv"))
    # plot_results
    plot_results(results, out_path)

    # generate text at inference
    start_token = dataset.vocab["<SOS>"]
    inputs, targets, preds = inference(transformer=transformer, test_dataset=test_dataset, start_token=start_token, temp=args.temp, test_samples=args.test_samples, logger=logger)
    inference_path = os.path.join(out_path, "inference")
    if not os.path.isdir(inference_path):
        os.makedirs(inference_path)
    with h5py.File(os.path.join(inference_path, "inference.h5"), 'w') as f:
        f.create_dataset('inputs', data=inputs)
        f.create_dataset('targets', data=targets)
        f.create_dataset('preds', data=preds)
    # for (path, arr) in zip([os.path.join(inference_path, "inputs.npy"), os.path.join(inference_path, "targets.npy"),
    #                         os.path.join(inference_path, "preds.npy")], [inputs, targets, preds]):
    #     np.save(path, arr)
    text_inputs = dataset.tokenizer.decode_batch(inputs.numpy())
    text_preds = dataset.tokenizer.decode_batch(preds.numpy())
    text_targets = dataset.tokenizer.decode_batch(targets.numpy())
    text_df = pd.DataFrame.from_records(
        dict(zip(["inputs", "targets", "preds"], [text_inputs, text_targets, text_preds])))
    text_df.to_csv(os.path.join(inference_path, "texts.csv"))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
