from build import *

with open('../assignment1-basics/TinyStoriesV2-GPT4-valid.txt', encoding='utf-8') as f:
    text = f.read()



tokenizer = BasicUztoken()

tokenizer.train(text, 10000,verbose=True)

print(tokenizer.vocab)
