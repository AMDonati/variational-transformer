import argparse
from src.data_provider.ROCDataset import ROCDataset
from src.models.transformer import Transformer, VAETransformer
from src.train.train_transformer import train, train_VAE
from src.train.utils import CustomSchedule, get_checkpoints, get_klweights
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

models = {"transformer": Transformer, "VAE": VAETransformer}
train_fn = {"transformer": train, "VAE": train_VAE}
train_losses = {"transformer": "train_loss", "VAE": "train_ce_loss"}  # to compute ppl.
val_losses = {"transformer": "val_loss", "VAE": "val_ce_loss"}


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    # parser.add_argument("-data_path", type=str, required=True, help="path for uploading the dataset")
    parser.add_argument("-max_samples", type=int, help="max samples for train dataset")
    # model parameters:
    parser.add_argument("-model", type=str, default="transformer", help="model: transformer or VAE for now.")
    parser.add_argument("-num_layers", type=int, default=1,
                        help="number of layers in the network.")
    parser.add_argument("-num_heads", type=int, default=1, help="number of attention heads for Transformer networks")
    parser.add_argument("-d_model", type=int, default=32, help="depth of attention parameters")
    parser.add_argument("-dff", type=int, default=32, help="dimension of feed-forward network")
    parser.add_argument("-pe", type=int, default=1000, help="maximum positional encoding")
    parser.add_argument("-p_drop", type=float, default=0.1, help="dropout on output layer")
    # VAE Transformer params:
    parser.add_argument("-latent", type=str, default="attention",
                        help="where to inject the latent in the VAE transformer")
    parser.add_argument("-simple_average", type=int, default=0,
                        help="simple average or average attention in the VAE encoder.")
    parser.add_argument("-beta_schedule", type=str, default="linear",
                        help="schedule for KL annealing (linear or linear with warm-up")
    parser.add_argument("-n_cycle", type=int, default=1,
                        help="number of cyclic schedule for KL annealing.")
    parser.add_argument("-beta_stop", type=float, default=1.,
                        help="maximum value for kl weights.")
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
        out_file = '{}_{}L_d{}_dff{}_pe{}_bs{}_pdrop{}'.format(args.model, args.num_layers, args.d_model, args.dff,
                                                               args.pe,
                                                               args.bs,
                                                               args.p_drop)
        if args.model == "VAE":
            out_file = out_file + "_{}".format(args.latent) + "_{}{}-{}".format(args.beta_schedule, args.n_cycle,
                                                                                args.beta_stop)
            if args.simple_average:
                out_file = out_file + "_simpleavg"
        datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_folder = os.path.join(args.output_path, out_file, datetime_folder)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder


def plot_results(results, out_path, train_key="train_loss", val_key="val_loss"):
    train_loss = results[train_key]
    val_loss = results[val_key]
    fig, ax = plt.subplots()
    x = np.linspace(1, len(train_loss), len(train_loss))
    ax.plot(x, train_loss, 'red', lw=2, label='train loss')
    ax.plot(x, val_loss, 'cyan', lw=2, label='val loss')
    ax.legend(fontsize=10)
    va = "bottom"
    for line in ax.lines:
        for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
            label = "{:.2f}".format(y_value)
            ax.annotate(label, (x_value, y_value), xytext=(0, 5),
                        textcoords="offset points", ha='center', va=va)
        va = "top"
    fig.savefig(os.path.join(out_path, "loss_plot.png"))


def run(args):
    # Create out path & logger & config.json file
    out_path = create_out_path(args)
    logger = create_logger(out_path)
    save_hparams(args, out_path)
    train_writer, val_writer = create_tensorboard_writer(out_path)

    # Load Dataset
    dataset = ROCDataset(data_path='data/ROC', batch_size=args.bs, max_samples=args.max_samples)
    train_data, val_data, test_data = dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = dataset.data_to_dataset(train_data, val_data, test_data)
    vocab_size = len(dataset.vocab)

    # Create Transformer
    transformer = models[args.model](
        num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads, dff=args.dff,
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        pe_input=args.pe, pe_target=args.pe, latent=args.latent, rate=args.p_drop, simple_average=args.simple_average)

    # Train Transformer
    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    checkpoint_path = create_ckpt_path(args, out_path)
    ckpt_manager = get_checkpoints(transformer, optimizer, checkpoint_path)

    # get range of kl_weights for KL annealing in VAE training:
    n_iter = args.ep * len(train_dataset)
    range_klweights = get_klweights(beta_schedule=args.beta_schedule, n_cycle=args.n_cycle, n_iter=n_iter,
                                    beta_stop=args.beta_stop)

    results = train_fn[args.model](EPOCHS=args.ep, train_dataset=train_dataset, val_dataset=val_dataset,
                                   transformer=transformer,
                                   optimizer=optimizer, loss_object=loss_object, ckpt_manager=ckpt_manager,
                                   logger=logger, train_writer=train_writer, val_writer=val_writer,
                                   range_klweights=range_klweights, tokenizer=dataset.tokenizer, out_path=out_path)
    results["train_ppl"] = np.exp(results[train_losses[args.model]])
    results["val_ppl"] = np.exp(results[val_losses[args.model]])
    # save results
    df_results = pd.DataFrame.from_records(results)
    df_results.to_csv(os.path.join(out_path, "train_history.csv"))
    # plot_results
    (train_key, val_key) = ("train_loss", "val_loss") if args.model == "transformer" else (
    "train_ce_loss", "val_ce_loss")
    plot_results(results, out_path, train_key, val_key)

    # generate text at inference
    start_token = dataset.vocab["<SOS>"]
    inputs, targets, preds = inference(transformer=transformer, test_dataset=test_dataset, start_token=start_token,
                                       temp=args.temp, test_samples=args.test_samples, logger=logger)
    inference_path = os.path.join(out_path, "inference")
    if not os.path.isdir(inference_path):
        os.makedirs(inference_path)
    with h5py.File(os.path.join(inference_path, "inference.h5"), 'w') as f:
        f.create_dataset('inputs', data=inputs)
        f.create_dataset('targets', data=targets)
        f.create_dataset('preds', data=preds)
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

# hparams for T-CVAE
# We set our model parameters based on preliminary experi- ments on the development data.
# For all models including baselines, dmodel is set to 512 and demb is set to 300. For Transformer models,
# the head of attention H is set to 8 and the number of Transformer blocks L is set to 6.
# The num- ber of LSTM layers is set to 2. For VAE models, dz is set to 64 and the annealing step is set to 20000.
# We apply dropout to the output of each sub-layer in Transformer blocks. We use a rate Pdrop = 0.15 for all models.
# We use the Adam Optimizer with an initial learning rate of 10−4, momentum β1 = 0.9, β2 = 0.99 and weight decay ? = 10−9.
# The batch size is set to 64. We use greedy search for all models and initialize them with 300-dimensional Glove word vectors.


# beta cycling schedule.
# Specifically, we use the cyclical schedule to anneal β for 10 periods (Fu et al., 2019).
# Within one period, there are three consecutive stages: Training AE (β = 0) for 0.5 proportion,
# annealing β from 0 to 1 for 0.25 proportion, and fixing β = 1 for 0.25 proportion.

# check influence of learning rate.
