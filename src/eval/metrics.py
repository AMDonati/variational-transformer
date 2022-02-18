from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config, OpenAIGPTTokenizer, TFOpenAIGPTLMHeadModel, OpenAIGPTConfig
import tensorflow as tf

gpt2_config = GPT2Config(vocab_size=50257)
gpt2_model = TFGPT2LMHeadModel(gpt2_config).from_pretrained("cache/gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_tokenizer.pad_token_id = [50256]

def gpt2_perplexity(sentence):
    inputs = gpt2_tokenizer(sentence, return_tensors="tf")
    outputs = gpt2_model(input_ids=inputs["input_ids"])
    logits = outputs["logits"]
    preds = logits[:, :-1, :]
    targets = inputs["input_ids"][:, 1:]
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    cross_entropy = tf.reduce_mean(ce(y_true=targets, y_pred=preds)) # (B,1,S)
    ppl = tf.math.exp(cross_entropy)
    return round(ppl.numpy(),2)

def gpt2_perplexity_batch(sentences):
    inputs = gpt2_tokenizer(sentences, padding=True, return_tensors="tf")
    outputs = gpt2_model(**inputs)
    preds_logits = outputs["logits"][:,:-1]
    labels = inputs["input_ids"][:,1:]
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = ce(y_true=labels, y_pred=preds_logits)
    ppl = tf.exp(tf.reduce_mean(loss, axis=-1))
    return tf.reduce_mean(ppl).numpy()