
import pickle
import regex




class Tokenizer:
    def __init__(self,  vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverted_vocab = {v:k for k,v in self.vocab.items()}
        self.merge_ranks = {pair3:ai for ai, pair3 in enumerate(merges)}
        # self.PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" #gpt4 pat

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #gpt-2

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        with open(vocab_filepath, 'rb') as fi:
            vocab = pickle.load(fi)


        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pat = "(" + "|".join(regex.escape(tok) for tok in self.special_tokens) + ")"
            docs = regex.split(split_pat, text)
        else: docs = [text]
        encoded_text = []
        for doc in docs:
            if self.special_tokens and doc in self.special_tokens:
                encoded_text.append(self.inverted_vocab[doc.encode("utf-8")])
                continue
            for match in regex.finditer(self.PAT, doc):

                word = [bytes([b]) for b in match.group().encode('utf-8')]
                while len(word) >= 2:
                    new_word = []
                    pairs = set(zip(word, word[1:]))
                    replace_pair = min(pairs, key=lambda p:self.merge_ranks.get(p,float("inf")))
                    if replace_pair not in self.merge_ranks:
                        break
                    merged = replace_pair[0] + replace_pair[1]
                    idxx = 0

                    while idxx < len(word):
                        if (idxx <len(word)-1) and   (word[idxx], word[idxx+1]) == replace_pair:
                            new_word.append(merged)
                            idxx+=2
                        else:

                            new_word.append(word[idxx])
                            idxx+=1
                    word = new_word

                encoded_text.extend([self.inverted_vocab[tok] for tok in word])

        return encoded_text

    def decode(self, ids: list[int]) -> str:
        decoded_text = ""
        allbytes = b"".join(self.vocab[idx] for idx in ids)
        decoded_text = allbytes.decode(errors='replace')
        return decoded_text

    from collections.abc import Iterable, Iterator
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id
