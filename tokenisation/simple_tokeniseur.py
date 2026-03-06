import re


# here we create a class to be able to encode and decode the text, meaning we can convert the text into a list of integers and vice versa.
class simpleTokeniseurV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        # E
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_word = sorted(set(preprocessed))
vocab_size = len(all_word)
enum_vocab = enumerate(all_word)
vocab = {token: integer for integer, token in enum_vocab}
tokenizer = simpleTokeniseurV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
