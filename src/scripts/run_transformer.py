import argparse
from src.data_provider.ROCDataset import ROCDataset
from src.models.transformer import Transformer
from src.train.train_transformer import train
from src.train.utils import CustomSchedule, get_checkpoints
import tensorflow as tf
import os
import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    #parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-max_samples", type=int, default=500, help="max samples for train dataset")
    # model parameters:
    parser.add_argument("-num_layers", type=int, default=1,
                        help="number of layers in the network. If == 0, corresponds to adding GPT2Decoder.")
    parser.add_argument("-num_heads", type=int, default=1, help="number of attention heads for Transformer networks")
    parser.add_argument("-d_model", type=int, default=8, help="depth of attention parameters")
    parser.add_argument("-dff", type=int, default=8, help="dimension of feed-forward network")
    parser.add_argument("-pe", type=int, default=50, help="maximum positional encoding")
    parser.add_argument("-p_drop", type=float, default=0.1, help="dropout on output layer")
    # training params.
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=5, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    # output_path params.
    parser.add_argument("-output_path", type=str, default="output", help="path for output folder")
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if loading ckpt)")
    # inference params.
    parser.add_argument("-past_len", type=int, default=4, help="number of timesteps for past timesteps at inference")
    parser.add_argument("-future_len", type=int, default=5,
                        help="number of predicted timesteps for multistep forecast.")
    parser.add_argument("-mc_samples", type=int, default=1, help="number of samples for MC Dropout algo.")
    parser.add_argument("-test_samples", type=int, help="number of test samples.")
    parser.add_argument("-temp", type=float, default=1., help="temperature for sampling text.")

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

    # Create out path & logger
    out_path = create_out_path(args)
    logger = create_logger(out_path)

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

    # save results
    df_results = pd.DataFrame.from_records(results)
    df_results.to_csv(os.path.join(out_path, "train_history.csv"))

    # plot_results
    plot_results(results, out_path)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
