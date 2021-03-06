import torch


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.allow_unk = False
        self.idx_to_token = dict(zip(list(vocab.values()), list(vocab.keys())))

    def encode(self, text, **kwargs):
        code = self.encode_(text, token_to_idx=self.vocab, allow_unk=self.allow_unk)
        if type(code) != torch.tensor and "return_tensors" in kwargs and kwargs["return_tensors"] == "pt":
            code = torch.tensor(code)
        return code

    def decode_batch(self, tensor, ignored=["<SOS>", "<PAD>"], stop_at_end=True):
        texts = []
        for b in range(tensor.shape[0]):
            texts.append(self.decode(tensor[b], ignored=ignored, stop_at_end=stop_at_end))
        return texts

    def decode(self, text, ignored=["<SOS>", "<PAD>"], stop_at_end=True):
        decode = self.decode_(text, idx_to_token=self.idx_to_token, stop_at_end=stop_at_end, delim=' ',
                              ignored=ignored)
        return decode

    def encode_(self, seq_tokens, token_to_idx, allow_unk):
        if type(seq_tokens) == str:
            seq_tokens = [seq_tokens]
        seq_idx = []
        for token in seq_tokens:
            if token not in token_to_idx:
                if allow_unk:
                    token = '<UNK>'
                else:
                    raise KeyError('Token "%s" not in vocab' % token)
            seq_idx.append(token_to_idx[token])
        return seq_idx

    def decode_(self, seq_idx, idx_to_token, stop_at_end, delim=' ', ignored=["<SOS>", "<PAD>"]):
        tokens = []
        for idx in seq_idx:
            token = idx_to_token[idx]
            if not token in ignored:
                if stop_at_end and token == '<EOS>':
                    break
                tokens.append(token)
        if delim is None:
            return tokens
        else:
            return delim.join(tokens)
