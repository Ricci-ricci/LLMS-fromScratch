import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

tokeniser = tiktoken.get_encoding("gpt2")


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokeniser, max_len, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokeniser.encode(text)

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_V1(
    text,
    batch_size=4,
    max_len=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokeniser = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokeniser, max_len, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )
    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_V1(
    raw_text, batch_size=1, max_len=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)


# let s try with a batch size greater than 1


# positional embedings
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_len = 4
dataloader = create_dataloader_V1(raw_text, batch_size=8, max_len=max_len, stride=4)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = token_embedding_layer(inputs)
context_len = max_len
pos_embedding_layer = torch.nn.Embedding(context_len, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_len))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
print(pos_embeddings.shape)
print(token_embeddings.shape)
