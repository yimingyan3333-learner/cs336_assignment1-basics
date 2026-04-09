from cs336_basics.tokenizer import BPETokenizer, encode_file_to_bin
from tests.test_tokenizer import test_encode_iterable_tinystories_sample_roundtrip
import random
import ipdb
import os

SEPARATOR = "<|endoftext|>"     # 文档分隔符（你常用的）
SAMPLE_NUM = 10
BLOCK_SIZE = 10*1024*1024

TINY_STORES = {
    'train_path': os.path.join("data", "TinyStoriesV2-GPT4-train.txt"),
    'valid_path': os.path.join("data", "TinyStoriesV2-GPT4-valid.txt"),
    'vocab_filepath': os.path.join("results", "tinystoires_train_vocab.pkl"),
    'merges_filepath': os.path.join("results", "tinystories_train_merges.pkl"),
    'out_bin_path_train': os.path.join("results", "tinystores_train_ids.bin"),
    'out_bin_path_valid': os.path.join("results", "tinystores_valid_ids.bin"),
}

OWT = {
    'train_path': os.path.join("data", "owt_train.txt"),
    'valid_path': os.path.join("data", "owt_valid.txt"),
    'vocab_filepath': os.path.join("results", "owt_train_vocab.pkl"),
    'merges_filepath': os.path.join("results", "owt_train_merges.pkl"),
    'out_bin_path_train': os.path.join("results", "owt_train_ids.bin"),
    'out_bin_path_valid': os.path.join("results", "owt_valid_ids.bin"),
}

def encode_toy_example():
    text = 'the cat ate'
    vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    bpe_tokenizer = BPETokenizer(vocab=vocab, merges=merges)
    print(bpe_tokenizer.encode(text))

def encode_tinystories_sample():
    boundaries = []
    current_start = 0
    leftover = b""
    sep_len = len(SEPARATOR)
    with open(TINY_STORES['valid_path'], "rb") as f:
        i = 0
        while True:
            print(i)
            block = f.read(BLOCK_SIZE)

            if not block:
                boundaries.append(current_start + f.tell())
                break
            text = leftover + block
            sep_index = text.find(SEPARATOR.encode("utf-8"))
            while (sep_index != -1): #chunk中存在分隔符
                current_end = current_start + sep_index + sep_len
                boundaries.append(current_end)
                current_start = current_end
                leftover = text[current_end:]
                sep_index = leftover.find(SEPARATOR.encode("utf-8"))
            else:
                leftover = text
            i += 1
    total_articles = len(boundaries) - 1
    print("total articles: ", total_articles)
    random_indices = random.sample(range(total_articles), 10)
    sample_docs = []
    for i in random_indices:
        start, end = boundaries[i], boundaries[i+1]
        with open("data/owt_valid.txt", "r", encoding="utf-8") as f:
            f.seek(start)
            text = f.read(end - start)
            sample_docs.append(text)
    print("sample docs: ", sample_docs)


    tokenizer = BPETokenizer.from_files(vocab_filepath="results/tinystoires_train_vocab.pkl", merges_filepath="results/tinystories_train_merges.pkl", special_tokens=["<|endoftext|>"])
    total_tokens = 0
    total_bytes = 0
    for doc in sample_docs:
        tokens = tokenizer.encode(doc)
        total_tokens += len(tokens)
        total_bytes += len(doc)
        print("token length: ", len(tokens), "total bytes: ", len(doc))
        print("compression ratio: ", len(doc)/len(tokens))
    print("total compression ratio: ", total_bytes/total_tokens)

def encode_and_store():
    data_param = OWT
    tokenizer = BPETokenizer.from_files(vocab_filepath=data_param["vocab_filepath"], merges_filepath=data_param["merges_filepath"], special_tokens=["<|endoftext|>"])
    encode_file_to_bin(tokenizer, data_param["train_path"], out_bin_path=data_param["out_bin_path_train"])
    encode_file_to_bin(tokenizer, data_param["valid_path"], out_bin_path=data_param["out_bin_path_valid"])




if __name__ == "__main__":
    encode_and_store()