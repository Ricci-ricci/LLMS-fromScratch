import re

# this is the step we follow to tokenise the text
# meaning we separate them into tokens, which are the smallest units of meaning in a text.
#
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_word = sorted(set(preprocessed))
vocab_size = len(all_word)
enum_vocab = enumerate(all_word)
vocab = {token: integer for integer, token in enum_vocab}
for i, items in enumerate(vocab.items()):
    print(items)
    if i > 50:
        break
