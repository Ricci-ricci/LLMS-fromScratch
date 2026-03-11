# Import libraries for tokenization and data handling
import tiktoken  # OpenAI's Byte Pair Encoding tokenizer
import torch  # PyTorch for tensor operations
from torch.utils.data import DataLoader, Dataset  # For batch processing

# Load GPT-2's tokenizer (uses Byte Pair Encoding)
# This tokenizer has a vocabulary of ~50,257 tokens
tokeniser = tiktoken.get_encoding("gpt2")


# Custom Dataset class for GPT training
# Creates input-target pairs with a sliding window approach
class GPTDatasetV1(Dataset):
    def __init__(self, text, tokeniser, max_len, stride):
        """
        Args:
            text: Raw text string to process
            tokeniser: Tokenizer to convert text to IDs
            max_len: Maximum sequence length (context window)
            stride: How many tokens to move the window forward each step
        """
        self.input_ids = []  # Will store input sequences
        self.target_ids = []  # Will store target sequences (shifted by 1)

        # Convert entire text to token IDs
        token_ids = tokeniser.encode(text)

        # Create overlapping chunks using a sliding window
        for i in range(0, len(token_ids) - max_len, stride):
            # Input: tokens from position i to i+max_len
            input_chunk = token_ids[i : i + max_len]
            # Target: tokens shifted by 1 (next token prediction)
            target_chunk = token_ids[i + 1 : i + max_len + 1]

            # Convert to PyTorch tensors and store
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return a single (input, target) pair by index"""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_V1(
    text,
    batch_size=4,  # Number of samples per batch
    max_len=256,  # Maximum sequence length (context window)
    stride=128,  # Window slide distance (128 = 50% overlap)
    shuffle=True,  # Randomize order for training
    drop_last=True,  # Drop incomplete final batch
    num_workers=0,  # Number of parallel workers for data loading
):
    """
    Creates a PyTorch DataLoader for GPT training.

    Args:
        text: Raw text string to process
        batch_size: How many sequences to process at once
        max_len: Length of each sequence
        stride: How much to shift the window (smaller = more overlap)
        shuffle: Whether to randomize sample order
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of subprocesses for data loading

    Returns:
        DataLoader object for iterating over batches
    """
    tokeniser = tiktoken.get_encoding("gpt2")
    # Create the dataset
    dataset = GPTDatasetV1(text, tokeniser, max_len, stride)
    # Wrap it in a DataLoader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )
    return dataloader


# ============================================================================
# EXAMPLE USAGE: Load text and create dataloaders
# ============================================================================

# Read the text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Create a simple dataloader (batch_size=1, small sequences for demonstration)
dataloader = create_dataloader_V1(
    raw_text, batch_size=1, max_len=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)  # Get first batch
second_batch = next(data_iter)  # Get second batch


# ============================================================================
# EXAMPLE: Batch size > 1
# ============================================================================

# Create dataloader with larger batch size
# batch_size=8 means we process 8 sequences simultaneously


# ============================================================================
# POSITIONAL EMBEDDINGS
# ============================================================================
# Language models need two types of embeddings:
# 1. Token embeddings: represent the meaning of each token
# 2. Positional embeddings: represent the position of each token in the sequence

vocab_size = 50257  # GPT-2 vocabulary size
output_dim = 256  # Embedding dimension (each token → 256D vector)

# Token embedding layer: converts token IDs to dense vectors
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_len = 4  # Maximum sequence length
# Create dataloader with batch_size=8
dataloader = create_dataloader_V1(raw_text, batch_size=8, max_len=max_len, stride=4)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)  # inputs shape: (8, 4) = 8 sequences of 4 tokens

# Convert token IDs to embeddings
# Shape: (batch_size, sequence_length, embedding_dim) = (8, 4, 256)
token_embeddings = token_embedding_layer(inputs)

# Positional embedding layer: adds position information
context_len = max_len
pos_embedding_layer = torch.nn.Embedding(context_len, output_dim)
# Create position indices [0, 1, 2, 3] and get their embeddings
# Shape: (4, 256) - one embedding per position
pos_embeddings = pos_embedding_layer(torch.arange(context_len))

# Combine token and positional embeddings (broadcasting adds pos to each sequence)
# Final shape: (8, 4, 256)
input_embeddings = token_embeddings + pos_embeddings

# Print shapes to verify
print(input_embeddings.shape)  # (8, 4, 256) - final input to the model
print(pos_embeddings.shape)  # (4, 256) - position embeddings
print(token_embeddings.shape)  # (8, 4, 256) - token embeddings
