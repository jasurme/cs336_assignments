from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing
import regex
from cs336_basics.utils import *
"""
Problem (train_bpe): BPE Tokenizer Training (15 points)
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:
Input
input_path: str Path to a text file with BPE tokenizer training data.

vocab_size: int A positive integer that defines the maximum final vocabulary size (including
the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

special_tokens: list[str] A list of strings to add to the vocabulary. During training, treat
them as hard boundaries that prevent merges across their spans, but do not include them when
computing merge statistics.

Your BPE training function should return the resulting vocabulary and merges:
Output
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the
vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list
item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation
"""

def process_chunk(args):
    path, start, end, special_tokens, PAT = args
    local_counts = {}
    with open(path,"rb")as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    split_pat = "|".join(regex.escape(tok) for tok in special_tokens)
    docs = regex.split(split_pat, chunk)
    for doc in docs:
        for match in regex.finditer(PAT, doc):
            piece = match.group()
            tokentuple = tuple(piece.encode('utf-8'))
            local_counts[tokentuple]=local_counts.get(tokentuple,0)+1

    return local_counts

def train_bpe(input_path:str, vocab_size:int, special_tokens: list[str])-> tuple[dict[int, bytes],list[tuple[bytes, bytes]] ]:
    num_processes=8
    PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" #gpt4 pat

    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #gpt2

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')

    tasks = [(input_path, start, end, special_tokens, PAT) for start, end in zip(boundaries[:-1], boundaries[1:])]


    with multiprocessing.Pool(num_processes) as pool:
        list_of_count_dicts= pool.map(process_chunk, tasks)

    word_counts = {} # dict of for example: (32, 114, 111, 121, 97, 108, 116, 121): 1
    for counts in list_of_count_dicts:
        for key, value in counts.items():
            word_counts[key] = word_counts.get(key, 0)+value

    vocab =  {i: bytes([i])  for i in range(256)} # {int: bytes}
    for idx, special_token in enumerate(special_tokens):
        vocab.update({256+idx: special_token.encode('utf-8')})


    pair_to_words = {}

    merges = [] # list[tuple(byte, byte)]
    pair_counts = {}
    for word_tuple,count in word_counts.items(): # word bytes tuple example: (32, 114, 111, 121, 97, 108, 116, 121)
        for a,b in zip(word_tuple, word_tuple[1:]):
            pair_to_words.setdefault((a,b), set()).add(word_tuple)
            pair_counts[(a,b)] = pair_counts.get((a,b), 0)+count

    base_tokensize = 256 + len(special_tokens)
    MERGE_COUNT = vocab_size - base_tokensize

    for i in range(MERGE_COUNT):
        new_idx = base_tokensize+i
        max_pair = max(pair_counts, key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))

        contained_words = list(pair_to_words[max_pair])
        for old_word in contained_words:
            cnt = word_counts[old_word]
            idxx = 0
            new_word = ()
            while idxx < len(old_word):
                if (idxx <len(old_word)-1) and   (old_word[idxx], old_word[idxx+1]) == max_pair:
                    new_word += (new_idx,)
                    idxx+=2
                else:

                    new_word +=(old_word[idxx],)
                    idxx+=1

            for pair in zip(old_word, old_word[1:]):
                if pair in pair_counts:
                    pair_counts[pair] -= cnt
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                s = pair_to_words.get(pair)
                if s is not None:
                    s.discard(old_word)
                    if not s:
                        del pair_to_words[pair]

            for pair1 in zip(new_word, new_word[1:]):
                pair_counts[pair1] = pair_counts.get(pair1, 0)+cnt
                pair_to_words.setdefault(pair1, set()).add(new_word)

            del word_counts[old_word]
            word_counts[new_word] = word_counts.get(new_word, 0) + cnt






        merges.append((vocab[max_pair[0]],vocab[max_pair[1]]))

        vocab[new_idx] = vocab[max_pair[0]]+ vocab[max_pair[1]]
        print(f'creating new vocab: {new_idx}: {vocab[max_pair[0]]+ vocab[max_pair[1]]}')

    return vocab, merges

import time
import tracemalloc
import resource
import pickle

if __name__ == "__main__":
    tracemalloc.start()
    start = time.perf_counter()

    vocab1, merges1 = train_bpe("owt_train.txt", 32000, ['<|endoftext|>'])

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ru_maxrss is in BYTES on macOS (you're on a Mac), KILOBYTES on Linux
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Peak memory (tracemalloc, main process): {peak / 1e6:.1f} MB")
    print(f"Peak RSS (OS, whole process): {peak_rss / 1e6:.1f} MB")  # macOS: /1e6

    with open('vocabs_openweb.pkl', 'wb') as filee:
        pickle.dump(vocab1, filee)
    with open("merges_openweb.pkl", "wb") as f:
        pickle.dump(merges1, f)
