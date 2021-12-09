'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import json
import numpy as np
from src.data_provider.tokenizer import Tokenizer
import tensorflow as tf
import os
import pandas as pd
import h5py

class ROCDataset:
    def __init__(self, data_path, batch_size=32, max_samples=None):
        self.data_path = data_path
        self.vocab_path = os.path.join(data_path, "vocab.json")
        self.batch_size = batch_size
        self.vocab = self.get_vocab()
        self.output_size = len(self.vocab)
        self.tokenizer = Tokenizer(self.vocab)
        self.name = "roc"
        self.max_samples = max_samples

    def get_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab

    def get_dataset_elements(self, dataset_path):
        dataset = pd.read_pickle(dataset_path)
        input_sentence = np.array([seq for seq in dataset.sentence1.values])
        target_sentence = np.array([seq for seq in dataset.sentence2.values])
        if self.max_samples is not None:
            input_sentence = input_sentence[:self.max_samples]
            target_sentence = target_sentence[:self.max_samples]
        return input_sentence, target_sentence

    def get_datasets(self):
        train_data = self.get_dataset_elements(os.path.join(self.data_path, "train_set.pkl"))
        val_data = self.get_dataset_elements(os.path.join(self.data_path, "val_set.pkl"))
        test_data = self.get_dataset_elements(os.path.join(self.data_path, "test_set.pkl"))
        return train_data, val_data, test_data

    def get_dataloader(self, data, batch_size):
        inputs, targets = data
        tfdataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets))
        tfdataloader = tfdataset.batch(batch_size, drop_remainder=True)
        return tfdataloader

    def data_to_dataset(self, train_data, val_data, test_data, num_dim=4):
        train_dataset = self.get_dataloader(train_data, self.batch_size)
        val_dataset = self.get_dataloader(val_data, self.batch_size)
        test_dataset = self.get_dataloader(test_data, self.batch_size)
        return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    dataset = ROCDataset(data_path='data/ROC')
    train_data, val_data, test_data = dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = dataset.data_to_dataset(train_data, val_data, test_data)
    print("done")
