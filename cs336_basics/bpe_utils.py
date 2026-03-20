import multiprocessing
from collections import Counter
import regex as re
import collections
import os
from typing import BinaryIO

from cs336_basics.utils import get_process_memory
from tqdm import tqdm
import time

import ipdb

PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenizer(chunk, special_tokens):
    '''
    chunk: str
    count_dict: dict[tuple[bytes], int]
    '''
    pattern = re.compile(PRETOKENIZER_PATTERN)
    docs = re.split(r'|'.join(map(re.escape, special_tokens)), chunk)
    count_dict = Counter()
    for doc in docs:
        for match in pattern.finditer(doc):
            key = tuple([bytes([byte]) for byte in match.group().encode('utf-8')])
            count_dict[key] += 1
    return count_dict


def process_single_chunk(args):
    chunk, special_tokens = args
    res = pretokenizer(chunk, special_tokens)
    return res


def split_dict_equally(origin_dict: dict, x: int) -> list[dict]:
    """
    将字典均匀拆分为x份，返回字典列表
    :param origin_dict: 待拆分的原字典
    :param x: 要拆分的份数
    :return: 拆分后的字典列表，长度为x
    """
    # 边界处理：x非正数抛异常，空字典返回空列表
    if x <= 0:
        raise ValueError("拆分份数x必须为正整数")
    if not origin_dict:
        return [{} for _ in range(x)]

    # 字典转有序键值对列表（保证拆分顺序）
    items = list(origin_dict.items())
    total = len(items)
    base = total // x  # 每份基础数量
    rem = total % x  # 余数（前rem份各多1个，保证均匀）

    result = []
    start = 0
    for i in range(x):
        # 前rem份取base+1个，剩余份取base个
        end = start + (base + 1) if i < rem else start + base
        # 切片转字典，添加到结果
        result.append(dict(items[start:end]))
        start = end
    return result


def byte_pairs_counting(word, count, byte_pairs, byte_pair_to_word_idx, idx):

    '''
    pretokens_counts: list[(tuple[bytes], int)]
    byte_pairs: dict[tuple[bytes], int]
    '''

    for i in range(len(word) - 1):
        byte_pairs[(word[i], word[i + 1])] += count
        byte_pair_to_word_idx[(word[i], word[i + 1])].add(idx)
    return byte_pairs, byte_pair_to_word_idx

def merge_sub_tuple(main_tuple, sub_tuple):
    n, m = len(main_tuple), len(sub_tuple)
    if m > n or m == 0:
        return main_tuple
    new_list = []
    i = 0
    while i < n:
        if main_tuple[i:i + m] == sub_tuple:
            new_list.append(b''.join(sub_tuple))
            i += m
        else:
            new_list.append(main_tuple[i])
            i += 1
    return tuple(new_list)


def bpe_merge(word_items, modified_word_idx, max_byte_pair):

    for idx in modified_word_idx:
        pretoken, count = word_items[idx]
        n = len(pretoken)

        i = 0
        new_pretoken = []
        while i < n:
            if i < n-1 and (pretoken[i], pretoken[i+1]) == max_byte_pair:
                new_pretoken.append(b''.join(max_byte_pair))
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1
        word_items[idx] = (tuple(new_pretoken), count)
    return word_items

def get_most_freq_byte_pair(byte_pairs):
    max_val = max(byte_pairs.values())
    candidates = [k for k, v in byte_pairs.items() if v == max_val]
    candidates.sort(reverse=True)
    return candidates[0]

def chunk_generator(input_path, boundaries, special_tokens):
    with open(input_path, "rb") as f:
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries) - 1,
                               desc="Processing chunks"):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk, special_tokens
def bpe_tokenizer(input_path, vocab_size, special_tokens, **kwargs):
    '''
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]
    '''
    print("The beginning of the program, memory usage：{} GB".format(get_process_memory(unit='GB')))
    num_bytes = 256
    # initialize vocabulary
    vocab = dict()
    for i in range(num_bytes):
        vocab[i] = bytes([i])
    for i, special_token in enumerate(special_tokens):
        vocab[num_bytes + i] = special_token.encode('utf-8')

    num_processes = kwargs.get('num_processes', multiprocessing.cpu_count())
    num_chunks = kwargs.get('num_chunks', multiprocessing.cpu_count())
    pretokens = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    with multiprocessing.Pool(num_processes) as pool:
        task_args = chunk_generator(input_path, boundaries, special_tokens)
        pool_res = tqdm(pool.imap_unordered(process_single_chunk, task_args), total=len(boundaries)-1, desc="pretokenization")
        for pretokens_ in pool_res:
            for k, v in pretokens_.items():
                pretokens[k] += v
    del boundaries
    ipdb.set_trace()

    print("After pretokenization, memory usage：{} GB".format(get_process_memory(unit='GB')))
    # calc byte pairs
    merges = []
    word_items = list(pretokens.items())
    byte_pairs = Counter()
    del pretokens

    byte_pair_to_word_idx = collections.defaultdict(set)
    modified_word_idx = None
    # print("word_items: ", word_items)

    for idx, (word, count) in enumerate(word_items):
        if len(word) < 2:
            continue
        byte_pairs, byte_pair_to_word_idx = byte_pairs_counting(word, count, byte_pairs, byte_pair_to_word_idx,
                                                                idx)
    for iter in tqdm(range(vocab_size - len(special_tokens) - num_bytes),
                  total=vocab_size - len(special_tokens) - num_bytes, desc="merging"):  # num of merges needs to run

        max_byte_pair = get_most_freq_byte_pair(byte_pairs)


        modified_word_idx = byte_pair_to_word_idx[max_byte_pair].copy()

        merges.append(max_byte_pair)

        new_vocab = b''.join(max_byte_pair)
        vocab[len(vocab)] = new_vocab

        # 1. 预先收集所有受影响单词的旧对子并减去计数
        for idx in modified_word_idx:
            word, count = word_items[idx]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                byte_pairs[pair] -= count
                # 只有当这个词以后不再包含这个 pair 时才移除索引
                # 但为了保险，这里先全部移除，后面合并后再重新加
                if idx in byte_pair_to_word_idx[pair]:
                    byte_pair_to_word_idx[pair].remove(idx)

        # 2. 统一合并
        word_items = bpe_merge(word_items, modified_word_idx, max_byte_pair)

        # 3. 统一加上新对子的计数
        for idx in modified_word_idx:
            new_word, count = word_items[idx]
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                byte_pairs[new_pair] += count
                byte_pair_to_word_idx[new_pair].add(idx)

        # 4. 强制清理当前已合并的对子，防止残留（由于浮点误差或重复计数）
        if max_byte_pair in byte_pairs:
            del byte_pairs[max_byte_pair]
        if max_byte_pair in byte_pair_to_word_idx:
            del byte_pair_to_word_idx[max_byte_pair]


    print("Finishing BPE, memory usage：{} GB".format(get_process_memory(unit='GB')))

    return vocab, merges

