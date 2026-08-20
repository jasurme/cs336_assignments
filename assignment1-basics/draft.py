from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing
import regex

PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
special_tokens = ['<|endoftext|>']

def process_chunk(args):
    path, start, end = args
    local_counts = {}
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    split_pat = "|".join(regex.escape(tok) for tok in special_tokens)
    docs = regex.split(split_pat, chunk)
    for doc in docs:
        for match in regex.finditer(PAT, doc):
            piece = match.group()
            tokentuple = tuple(piece.encode('utf-8'))
            local_counts[tokentuple] = local_counts.get(tokentuple, 0) + 1
    return local_counts


if __name__ == "__main__":
    num_processes = 32
    input_path = "TinyStoriesV2-GPT4-valid.txt"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')

    tasks = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(num_processes) as pool:
        list_of_count_dicts = pool.map(process_chunk, tasks)

    print(len(list_of_count_dicts))
