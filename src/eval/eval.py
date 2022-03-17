import tensorflow as tf
import pandas as pd
import time


def inference(transformer, test_dataset, start_token, max_len=21, decoding="sampling", temp=1, test_samples=-1, logger=None, training=False):
    start_time = time.time()
    all_preds, all_targets, all_inputs = [], [], []
    if test_samples is None:
        test_samples = -1
    for (inputs, targets) in test_dataset.take(test_samples):
        #print("decoding text sample")
        tar_inp = tf.constant([start_token], shape=(inputs.shape[0], 1), dtype=tf.int32)
        for i in range(max_len):
            #print("decoding {}-th word".format(i+1))
            predictions, _, _, _ = transformer((inputs, tar_inp),
                                         training=training) #TODO: choose if training=True or False.
            last_pred = predictions[:, -1]
            if decoding == "sampling":
                last_pred = tf.random.categorical(logits=last_pred / temp, num_samples=1, dtype=tf.int32)
            elif decoding == "greedy":
                last_pred = tf.expand_dims(tf.math.argmax(last_pred, axis=-1, output_type=tf.int32),
                                           axis=-1)
            tar_inp = tf.concat([tar_inp, last_pred], axis=-1)
        all_preds.append(tar_inp)
        all_targets.append(targets)
        all_inputs.append(inputs)
    all_preds = tf.stack(all_preds, axis=0)
    all_targets = tf.stack(all_targets, axis=0)
    all_inputs = tf.stack(all_inputs, axis=0)
    if logger is not None:
        logger.info("TIME FOR INFERENCE:{}".format(time.time() - start_time))
    return tf.reshape(all_inputs, shape=(-1, all_inputs.shape[-1])), tf.reshape(all_targets, shape=(-1, all_targets.shape[-1])), tf.reshape(all_preds, shape=(
    -1, all_preds.shape[-1]))

def inference_multisentence(transformer, test_dataset, start_token, max_len=21, temp=1, test_samples=-1, training=False, num_samples=10, decoding="sampling"):
    all_preds, all_targets, all_inputs = [], [], []
    if test_samples is None:
        test_samples = -1
    for (inputs, targets) in test_dataset.take(test_samples):
        tar_inp = tf.constant([start_token], shape=(inputs.shape[0], 1), dtype=tf.int32)
        for i in range(max_len):
            num_samples_ = num_samples if i == 0 else 1
            if i == 1:
                inputs = tf.repeat(inputs, repeats=[num_samples]*inputs.shape[0], axis=0)
            predictions, _, _, _ = transformer((inputs, tar_inp),
                                         training=training)
            last_pred = predictions[:, -1]
            if decoding == "sampling":
                last_pred = tf.random.categorical(logits=last_pred / temp, num_samples=num_samples_, dtype=tf.int32) # shape (B, num_samples) or (B*num_samples)
            else:
                last_preds = []
                for s in range(num_samples_):
                    last_preds.append(tf.math.argmax(last_pred, axis=-1, output_type=tf.int32))
                last_pred = tf.stack(last_preds, axis=-1)
            last_pred = tf.reshape(last_pred, shape=(-1,1)) # shape (B*num_samples, 1)
            tar_inp = tf.repeat(tar_inp, repeats=[num_samples_]*tar_inp.shape[0], axis=0)
            tar_inp = tf.concat([tar_inp, last_pred], axis=-1)
        all_preds.append(tar_inp)
        all_targets.append(tf.repeat(targets, repeats=[num_samples]*targets.shape[0], axis=0))
        all_inputs.append(inputs)
    all_preds = tf.stack(all_preds, axis=0)
    all_targets = tf.stack(all_targets, axis=0)
    all_inputs = tf.stack(all_inputs, axis=0)

    return tf.reshape(all_inputs, shape=(-1, all_inputs.shape[-1])), tf.reshape(all_targets, shape=(-1, all_targets.shape[-1])), tf.reshape(all_preds, shape=(
    -1, all_preds.shape[-1]))


if __name__ == '__main__':
    from src.models.transformer import Transformer, VAETransformer
    from src.data_provider.ROCDataset import ROCDataset

    # Load Dataset
    dataset = ROCDataset(data_path='data/ROC', batch_size=32, max_samples=1000)
    train_data, val_data, test_data = dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = dataset.data_to_dataset(train_data, val_data, test_data)

    transformer = Transformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=len(dataset.vocab), target_vocab_size=len(dataset.vocab),
        pe_input=10000, pe_target=6000)

    start_token = dataset.vocab["<SOS>"]

    inputs, targets, preds = inference(transformer=transformer, test_dataset=test_dataset, start_token=start_token,
                                       max_len=10, test_samples=5)
    print(preds.shape)
    print(targets.shape)

    text_preds = dataset.tokenizer.decode_batch(preds.numpy())
    text_targets = dataset.tokenizer.decode_batch(targets.numpy())
    text_df = pd.DataFrame.from_records(dict(zip(["targets", "preds"], [text_targets, text_preds])))
    print("done for baseline Transformer")

    # Evaluating multisentence inference
    inputs, targets, preds = inference_multisentence(transformer=transformer, test_dataset=test_dataset, start_token=start_token,
                                       max_len=10, test_samples=10, decoding="greedy")
    print("preds shape", preds.shape)
    print("targets shape", targets.shape)

    print("------------------------------------------- evaluating VAE Transformer-------------------------------------------------")

    # evaluating on VAE Transformer
    vae_transformer = VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=len(dataset.vocab), target_vocab_size=len(dataset.vocab),
        pe_input=10000, pe_target=6000, latent="output")

    start_token = dataset.vocab["<SOS>"]

    inputs, targets, preds = inference(transformer=vae_transformer, test_dataset=test_dataset, start_token=start_token,
                                       max_len=10, test_samples=5)

    print(preds.shape)
    print(targets.shape)

    text_preds = dataset.tokenizer.decode_batch(preds.numpy())
    text_targets = dataset.tokenizer.decode_batch(targets.numpy())
    text_df = pd.DataFrame.from_records(dict(zip(["targets", "preds"], [text_targets, text_preds])))
    print("done for VAE Transformer")


