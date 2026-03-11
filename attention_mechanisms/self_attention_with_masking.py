import torch  # PyTorch for tensor operations
import torch.nn as nn  # Neural network modules

# ============================================================================
# SELF-ATTENTION V1: Using nn.Parameter (Manual Approach)
# ============================================================================
# This implementation uses nn.Parameter to create learnable weight matrices.
# nn.Parameter wraps tensors and marks them as trainable parameters.


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        """
        Initialize self-attention layer with trainable parameter matrices.

        Args:
            d_in: Input dimension (embedding size)
            d_out: Output dimension (projection size)
        """
        super().__init__()
        self.d_out = d_out

        # Create three learnable weight matrices for Q, K, V projections
        # nn.Parameter makes these tensors trainable during optimization
        # torch.rand initializes with random values in [0, 1)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))  # Query projection
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))  # Key projection
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))  # Value projection

    def forward(self, x):
        """
        Compute self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
               or (seq_len, d_in) for single sequence

        Returns:
            Context vectors of shape (batch_size, seq_len, d_out) or (seq_len, d_out)
        """
        # Project inputs to keys, queries, and values using matrix multiplication
        # @ operator performs matrix multiplication
        keys = x @ self.W_key  # Shape: (..., seq_len, d_out)
        queries = x @ self.W_query  # Shape: (..., seq_len, d_out)
        values = x @ self.W_value  # Shape: (..., seq_len, d_out)

        # Compute attention scores (similarity between queries and keys)
        # .T transposes the last two dimensions
        attn_scores = queries @ keys.T  # Shape: (..., seq_len, seq_len)

        # Apply scaled dot-product attention
        # Scale by sqrt(d_k) to prevent gradients from vanishing
        # Apply softmax to convert scores to probabilities
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Compute context vectors as weighted sum of values
        context_vector = attn_weights @ values  # Shape: (..., seq_len, d_out)
        return context_vector


# ============================================================================
# SELF-ATTENTION V2: Using nn.Linear (High-Level Approach)
# ============================================================================
# This implementation uses nn.Linear layers instead of raw Parameter matrices.
# nn.Linear is more convenient and handles weight initialization automatically.


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        Initialize self-attention layer with Linear layers.

        Args:
            d_in: Input dimension (embedding size)
            d_out: Output dimension (projection size)
            qkv_bias: Whether to include bias terms in Q, K, V projections
                     (typically False in Transformers)
        """
        super().__init__()
        self.d_out = d_out

        # Create three Linear layers for Q, K, V projections
        # nn.Linear automatically initializes weights and optionally adds bias
        # This is the standard approach in modern Transformers
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        Compute self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
               or (seq_len, d_in) for single sequence

        Returns:
            Context vectors of shape (batch_size, seq_len, d_out) or (seq_len, d_out)
        """
        # Project inputs by calling Linear layers as functions
        # NOTE: With nn.Linear, we CALL the layer, not use @ operator
        keys = self.W_key(x)  # Shape: (..., seq_len, d_out)
        queries = self.W_query(x)  # Shape: (..., seq_len, d_out)
        values = self.W_value(x)  # Shape: (..., seq_len, d_out)

        # Compute attention scores (similarity between queries and keys)
        attn_scores = queries @ keys.T  # Shape: (..., seq_len, seq_len)

        # Apply scaled dot-product attention with softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Compute context vectors as weighted sum of values
        context_vector = attn_weights @ values  # Shape: (..., seq_len, d_out)
        return context_vector


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

# Example sentence: "Your journey starts with one step"
# Each word represented as a 3-dimensional embedding vector
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],  # step (x^6)
    ]
)
d_in = inputs.shape[1]  # Input dimension: 3
d_out = 2  # Output dimension: 2 (projecting to lower dimension)

# Create instances of both attention implementations
torch.manual_seed(123)  # Set random seed for reproducibility
sa_v1 = SelfAttention_v1(d_in, d_out)

torch.manual_seed(789)  # Different seed for v2 to show different behavior
sa_v2 = SelfAttention_v2(d_in, d_out)

# ============================================================================
# CAUSAL MASKING (for autoregressive/decoder models like GPT)
# ============================================================================
# In language modeling, we want to prevent tokens from "looking ahead" at future tokens.
# This is crucial for training models to predict the next token.

# Compute queries, keys for demonstration
queries = sa_v2.W_query(inputs)  # Shape: (6, 2)
keys = sa_v2.W_key(inputs)  # Shape: (6, 2)
attn_scores = queries @ keys.T  # Shape: (6, 6) - all pairwise scores
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)

# Get sequence length for masking
context_length = attn_scores.shape[0]  # 6 tokens

# ============================================================================
# METHOD 1: Simple Masking (multiply by lower triangular matrix)
# ============================================================================
# Create a lower triangular mask (1s below and on diagonal, 0s above)
# torch.tril creates a lower triangular matrix:
# [[1, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0],
#  [1, 1, 1, 0, 0, 0],
#  [1, 1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1, 1]]
mask_simple = torch.tril(torch.ones(context_length, context_length))

# Multiply attention weights by mask (sets future positions to 0)
masked_simple = attn_weights * mask_simple

# Renormalize so each row sums to 1 again
# dim=1: sum across columns for each row
# keepdim=True: keeps the dimension as (6, 1) instead of (6,) for broadcasting
row_sums = masked_simple.sum(dim=1, keepdim=True)  # Shape: (6, 1)
masked_simple_norm = masked_simple / row_sums  # Normalize each row

# ============================================================================
# METHOD 2: Masking with -inf (preferred approach)
# ============================================================================
# This is the standard method used in practice because it's more numerically stable.
# We mask BEFORE softmax by setting future positions to -infinity.

# Create upper triangular mask (1s above diagonal, 0s elsewhere)
# torch.triu with diagonal=1 excludes the diagonal:
# [[0, 1, 1, 1, 1, 1],
#  [0, 0, 1, 1, 1, 1],
#  [0, 0, 0, 1, 1, 1],
#  [0, 0, 0, 0, 1, 1],
#  [0, 0, 0, 0, 0, 1],
#  [0, 0, 0, 0, 0, 0]]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

# Replace masked positions (1s) with -infinity in attention scores
# .bool() converts to boolean mask
# -torch.inf ensures these positions get ~0 probability after softmax
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# Apply softmax to masked scores
# The -inf values become 0 after softmax, effectively preventing attention to future tokens
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

# Result: Each token can only attend to itself and previous tokens
# Token 0: attends to [0]
# Token 1: attends to [0, 1]
# Token 2: attends to [0, 1, 2]
# etc.

# ============================================================================
# DROPOUT: Regularization technique
# ============================================================================
# In practice, we also apply dropout to attention weights to prevent overfitting
# Example: attn_weights = nn.Dropout(p=0.1)(attn_weights)
torch.manual_seed(123)
droupout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
