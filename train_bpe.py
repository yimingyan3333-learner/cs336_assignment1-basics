import time

import json
import os
import pickle
from cs336_basics import bpe_utils
import pathlib
from functools import lru_cache
import ipdb
from cs336_basics.utils import get_process_memory

def train_bpe_tinystories():
    time_start = time.time()
    input_path = os.path.join("data", "TinyStoriesV2-GPT4-train.txt")
    vocab, merges = bpe_utils.bpe_tokenizer(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    time_end = time.time()
    print("Time taken: ", (time_end - time_start))
    max_word_len = 0
    max_len_word = ""
    for w in vocab.values():
        if len(w) > max_word_len:
            max_word_len = len(w)
            max_len_word = w
    print("Max length word: {} with lengh {}".format(max_len_word, max_word_len))
    with open(os.path.join("results", "tinystoires_train_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join("results", "tinystories_train_merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

def train_bpe_expts_owt():
    time_start = time.time()
    input_path = os.path.join("data", "owt_train.txt")
    vocab, merges = bpe_utils.bpe_tokenizer(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_chunks=100,
        num_processors=5,
    )
    time_end = time.time()
    print("Time taken: ", (time_end - time_start))
    max_word_len = 0
    max_len_word = ""
    for w in vocab.values():
        if len(w) > max_word_len:
            max_word_len = len(w)
            max_len_word = w
    print("Max length word: {} with lengh {}".format(max_len_word, max_word_len))

    with open(os.path.join("results", "owt_train_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join("results", "owt_train_merges.pkl"), "wb") as f:
        pickle.dump(merges, f)



if __name__ == "__main__":
    main_pid = os.getpid()
    print(f"主进程PID：{main_pid}")
    train_bpe_tinystories()
