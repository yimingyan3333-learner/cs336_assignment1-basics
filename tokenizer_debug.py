from cs336_basics.tokenizer import BPETokenizer
from tests.test_tokenizer import test_encode_iterable_tinystories_sample_roundtrip
import ipdb

def encode_toy_example():
    text = 'the cat ate'
    vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    bpe_tokenizer = BPETokenizer(vocab=vocab, merges=merges)
    print(bpe_tokenizer.encode(text))

if __name__ == "__main__":
    test_encode_iterable_tinystories_sample_roundtrip()