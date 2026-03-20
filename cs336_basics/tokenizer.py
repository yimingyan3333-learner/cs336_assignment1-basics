from logging import setLoggerClass

from math import inf

import pickle
import regex as re
import ipdb

PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token.encode('utf-8') not in self.vocab.values():
                    self.vocab[len(vocab)] = special_token
        self.vocab2idx = {k: v for v, k in self.vocab.items()}
        self.merges2idx = dict(zip(merges, range(len(merges))))
        self.idx2merges = dict(zip(range(len(merges)), merges))
        self.special_tokens = special_tokens
        self.special_tokens_bytes = None
        if self.special_tokens is not None:
            self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        '''
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens=special_tokens)


    def pretokenize(self, text) -> list[tuple[bytes]]:
        pattern = re.compile(PRETOKENIZER_PATTERN)
        if self.special_tokens != None:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = r"|".join(map(re.escape, special_tokens_sorted))
            docs = re.split(f"({split_pattern})", text)
        else:
            docs = [text]
        pretokenized = []
        for doc in docs:
            if self.special_tokens != None and doc in self.special_tokens:
                pretokenized.append(doc.encode('utf-8'))
                continue
            for match in pattern.finditer(doc):
                pretokenized.append(tuple([bytes([byte]) for byte in match.group().encode('utf-8')]))
        return pretokenized

    def merge_fn(self, word_tuple, merge_pair):
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple)-1 and (word_tuple[i], word_tuple[i+1]) == merge_pair:
                new_word.append(b''.join(merge_pair))
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        return tuple(new_word)


    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        pretokenized = self.pretokenize(text)
        encoded_text = []
        for word_tuple in pretokenized:
            if self.special_tokens_bytes is not None and word_tuple in self.special_tokens_bytes:
                encoded_text.append(self.vocab2idx[word_tuple])
                continue

            while True:
                idx_min = len(self.vocab)
                for i in range(len(word_tuple)-1):

                    if (word_tuple[i], word_tuple[i+1]) in self.merges2idx:
                        idx = self.merges2idx[(word_tuple[i], word_tuple[i+1])]
                        if idx < idx_min:
                            idx_min = idx
                if idx_min < len(self.vocab):
                    next_merge = self.idx2merges[idx_min]
                    word_tuple = self.merge_fn(word_tuple, next_merge)
                else:
                    break # cannot find merges, finish encoding
            for token in word_tuple:
                if token not in self.vocab2idx:
                    print("token {} not in vocab".format(token))
                else:
                    encoded_text.append(self.vocab2idx[token])
        return encoded_text

    def encode_iterable(self, iterable):
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        '''
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        output_bytes = b""
        for id in ids:
            output_bytes += self.vocab[id]
        output = output_bytes.decode('utf-8', errors='replace')
        return output