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


class simpleTokeniseurV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        # E
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_word = sorted(set(preprocessed))
vocab_size = len(all_word)
enum_vocab = enumerate(all_word)
vocab = {token: integer for integer, token in enum_vocab}
tokenizer = simpleTokeniseurV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
pride."""
ids = tokenizer.encode(text)

text = "Hello, do you like tea?"
# to handle the fact that some tokens may not be in the vocab, we can add a special token for unknown tokens, and a special token for end of text.
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# exemple using simpleTokenizerV2
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text3 = " <|endoftext|> ".join((text1, text2))
tokenize2 = simpleTokeniseurV2(vocab)
ids = tokenize2.encode(text3)
token2 = tokenize2.decode(ids)
print(ids)
print(token2)
