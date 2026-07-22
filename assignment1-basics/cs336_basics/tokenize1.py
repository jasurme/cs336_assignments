import pickle
import array
from encode_decode import Tokenizer

# 1. Load your variables
with open('vocabs.pkl', 'rb') as f:
    vocabs = pickle.load(f)

with open('merges.pkl', 'rb') as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(vocabs, merges, ['<|endoftext|>'])

# 2. Stream directly to binary file in chunks
# 'H' stands for unsigned 16-bit integers (uint16)
token_buffer = array.array('H')
chunk_size = 500_000  # Flush to disk every 500k tokens

print("Starting streaming to disk...")

with open("../TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f_in, \
     open("val_tinystories.bin", "wb") as f_out:

    for token_id in tokenizer.encode_iterable(f_in):
        token_buffer.append(token_id)

        # Periodically dump to disk to keep RAM perfectly flat
        if len(token_buffer) >= chunk_size:
            token_buffer.tofile(f_out)
            token_buffer = array.array('H')  # Reset buffer safely

    # Don't forget to flush the last remaining tokens!
    if token_buffer:
        token_buffer.tofile(f_out)

print('done ✅')
