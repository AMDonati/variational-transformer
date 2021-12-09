import json
import numpy as np
from nltk import word_tokenize
import pandas as pd
import re


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def get_sentences(df, max_samples=None):
    df["sentence_1_2"] = df.sentence1 + " " + df.sentence2
    sentences = df.sentence_1_2
    sentences_1, sentences_2 = df["sentence1"], df["sentence2"]
    if max_samples is not None:
        sentences = sentences[:max_samples]
        sentences_1 = sentences_1[:max_samples]
        sentences_2 = sentences_2[:max_samples]
    return sentences, sentences_1, sentences_2


def split_train_test(sentences, val_size=5000, test_size=5000):
    train_size = len(sentences) - (val_size + test_size)
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:train_size + val_size]
    test_sentences = sentences[train_size + val_size:train_size + val_size + test_size]
    paths = ["data/ROC/train_set.pkl", "data/ROC/val_set.pkl", "data/ROC/test_set.pkl"]
    for df, path in zip([train_sentences, val_sentences, test_sentences], paths):
        df.to_pickle(path)
    print("saving dataset splits in pkl files...")
    return train_sentences, val_sentences, test_sentences

def tokenize(sentences, vocab):
    tokenize_func = lambda t: word_tokenize(t)
    tok_to_id_func = lambda t: [vocab["<SOS>"]]+[vocab[w] for w in t if w in vocab.keys()]+[vocab["<EOS>"]]
    #TODO: add a SOS and EOS token ?
    tokenized_sentences = sentences.apply(tokenize_func)
    tokens_id = tokenized_sentences.apply(tok_to_id_func)
    len_sentences = tokens_id.apply(len)
    max_len = np.max(len_sentences)
    print("max seq len", max_len)
    pad_func = lambda t: t + [0] * (max_len - len(t))
    padded_sentences = tokens_id.apply(pad_func)
    return padded_sentences, len_sentences

# def encode(lang1, lang2):
#     lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
#         lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
#
#     lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
#         lang2.numpy()) + [tokenizer_en.vocab_size + 1]
#
#     return lang1, lang2

# def filter_max_length(x, y, max_length=MAX_LENGTH):
#   return tf.logical_and(tf.size(x) <= max_length,
#                         tf.size(y) <= max_length)


def clean_text(sentences):
    clean_func1 = lambda t: ' '.join(t.split("-"))
    clean_func2 = lambda t: ' '.join(re.split(r"([0-9]+)([a-z]+)", t, re.I))
    clean_func3 = lambda t: ' '.join(re.split(r"([a-z]+)([0-9]+)", t, re.I))
    clean_func4 = lambda t: t.lower().replace("&", "and")
    sentences = sentences.apply(clean_func1)
    sentences = sentences.apply(clean_func2)
    sentences = sentences.apply(clean_func3)
    sentences = sentences.apply(clean_func4)
    return sentences


def get_vocab(sentences, tokens_to_remove=["$", "%", "'", "''"], special_tokens=["<PAD>", "<SOS>", "<EOS>"]): #TODO: add special tokens !
    print("Building vocab....")
    tokenize_func = lambda t: word_tokenize(t.lower())
    # tokens = word_tokenize(' '.join(sentences))
    tokenized_sentences = sentences.apply(tokenize_func)
    tokenized_sentences = tokenized_sentences.values
    tokens = [w for s in tokenized_sentences for w in s]
    unique_tokens = list(set(tokens))
    for token in tokens_to_remove:
        unique_tokens.remove(token)
    unique_tokens.sort()
    vocab = {v: k for k, v in enumerate(special_tokens + unique_tokens)}
    print("vocab length:", len(vocab))
    print("saving vocab...")
    with open("data/ROC/vocab.json", "w") as f:
        json.dump(vocab, f)
    return tokens, vocab

    # words to remove.
    # "$": 4509, "%": 7129, "&": 534, "'": 823, "''": 9236,


def preprocess_data(data_path):
    df = load_data(data_path)
    sentences, sentences_1, sentences_2 = get_sentences(df)
    sentences, sentences_1, sentences_2 = clean_text(sentences), clean_text(sentences_1), clean_text(sentences_2)
    tokens, vocab = get_vocab(sentences)
    input_sentences, _ = tokenize(sentences_1, vocab)
    target_sentences, _ = tokenize(sentences_2, vocab)
    data = pd.concat([input_sentences, target_sentences], axis=1)
    path = "data/ROC/data.pkl"
    data.to_pickle(path)
    train_sentences, val_sentences, test_sentences = split_train_test(data)
    return train_sentences, val_sentences, test_sentences


if __name__ == '__main__':
    data_path = "data/ROC/ROCStories_winter2017.csv"
    train_sentences, val_sentences, test_sentences = preprocess_data(data_path)
    print("done")

    # #Thelatest
    # release
    # includes
    # 98, 159
    # ROCStories and 3, 744
    # Story
    # Cloze
    # Test
    # instances
