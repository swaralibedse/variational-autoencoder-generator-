import re

class SmilesTokenizer:
    def __init__(self, smiles_list, pad_token='[PAD]', unk_token='[UNK]'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        charset = sorted(set(''.join(smiles_list)))
        self.tokens = [pad_token, unk_token] + charset
        self.token2idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    @property
    def vocab_size(self):
        return len(self.tokens)

    @property
    def pad_token_idx(self):
        return self.token2idx[self.pad_token]

    def encode(self, smiles, max_len=120):
        return [self.token2idx.get(c, self.token2idx[self.unk_token]) for c in smiles[:max_len]]

    def decode(self, indices):
        return ''.join([self.idx2token.get(i, self.unk_token) for i in indices if i != self.pad_token_idx])
