from utils import *
import regex as re
class BasicUztoken:
    def __init__(self):
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.merges = {} # (int, int) -> int
        self.special_tokens = {'<|endoftext|>':100001} # str -> int, e.g {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # {int -> bytes}

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0] +  vocab[p1]
        for special,idxx in self.special_tokens.items():  # some error here in logic? why we should give idxx, isnt it hardcoding
            vocab[idxx] = special.encode('utf-8')
        return vocab


    def train(self, text, vocab_size=3000, verbose=False):
        if vocab_size <256:
            raise ValueError('vocab size should be at least 256...')

        MERGING_COUNT = vocab_size - 256-len(self.special_tokens)
        print('starting training...')
        ids=[]
        for match in re.finditer(self.pattern,text):
            ids.append(list(match.group().encode('utf-8')))

        vocab = {idx:bytes([idx]) for idx in range(256)}
        merges= {}

        for i in range(MERGING_COUNT):
            stats = {}
            for chunk_ids in ids:

                get_stats(ids=chunk_ids,countss= stats)

            max_pair = max(stats, key=stats.get)

            new_id = 256+i
            ids = [merge(chunk_ids1,max_pair,new_id) for chunk_ids1 in ids]
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]]+vocab[max_pair[1]]
            if verbose:

                print(f"{i+1}/{MERGING_COUNT}: {max_pair}-> {new_id}:          [{vocab[max_pair[0]].decode('utf-8',errors='replace')}][{vocab[max_pair[1]].decode('utf-8',errors='replace')}]->[{vocab[new_id].decode('utf-8',errors='replace')}]")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids: list):
        text_bytes = ""
        for idx in ids:
            text_bytes =b"".join(self.vocab[idx])
        text = text_bytes.decode('utf-8', errors='replace')
        return text
#merges: (int, int)-> int
#vocab: int -> bytes
    def encode(self,text):
        tokens = list(text.encode('utf-8'))
        while True:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p:self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
