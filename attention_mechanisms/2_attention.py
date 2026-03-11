import torch  # PyTorch for tensor operations

# ============================================================================
# QUERY-KEY-VALUE ATTENTION WITH TRAINABLE WEIGHTS
# ============================================================================
# This demonstrates a more advanced attention mechanism where we learn
# separate transformations for Queries, Keys, and Values.
# This is the approach used in real Transformers (BERT, GPT, etc.)

# Example sentence: "Your journey starts with one step"
# Each word is represented as a 3-dimensional embedding vector
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

# ============================================================================
# STEP 1: Define Learnable Weight Matrices (Q, K, V Projections)
# ============================================================================
# Instead of using embeddings directly, we project them through learned matrices.
# This gives the model flexibility to learn what information is important.

x_2 = inputs[1]  # Select word "journey" for single-word example
d_in = inputs.shape[1]  # Input dimension: 3 (embedding size)
d_out = 2  # Output dimension: 2 (we're projecting to lower dimension)

# Create three trainable weight matrices
# nn.Parameter makes these tensors learnable during training
# requires_grad=False means we won't update them in this example (just demonstration)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # (3, 2)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # (3, 2)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # (3, 2)

# ============================================================================
# STEP 2: Project Single Word Through Weight Matrices
# ============================================================================
# Transform the "journey" embedding using the weight matrices
query_2 = x_2 @ W_query  # Shape: (3,) @ (3, 2) = (2,) - Query for "journey"
key_2 = inputs @ W_key  # Shape: (6, 3) @ (3, 2) = (6, 2) - Keys for all words
value_2 = inputs @ W_value  # Shape: (6, 3) @ (3, 2) = (6, 2) - Values for all words

# ============================================================================
# STEP 3: Compute Keys and Values for All Words
# ============================================================================
# Project all input embeddings to get keys and values
keys = inputs @ W_key  # (6, 3) @ (3, 2) = (6, 2) - All keys
values = inputs @ W_value  # (6, 3) @ (3, 2) = (6, 2) - All values

# ============================================================================
# STEP 4: Compute Attention Scores
# ============================================================================
# Calculate similarity between query and keys

# Example: Single score (query "journey" with its own key)
keys_2 = keys[1]  # Key for "journey"
attn_score_22 = query_2.dot(keys_2)  # Dot product: scalar similarity score
# Compute scores for query "journey" with ALL keys
# @ operator is matrix multiplication
# keys.T transposes keys from (6, 2) to (2, 6)
# query_2: (2,) @ keys.T: (2, 6) = (6,) - 6 scores, one per word
attn_scores_2 = query_2 @ keys.T

# ============================================================================
# STEP 5: Scaled Dot-Product Attention
# ============================================================================
# Scale the scores by sqrt(d_k) to prevent gradients from becoming too small
# This is a key innovation in the Transformer architecture

d_k = keys.shape[-1]  # Dimension of key vectors: 2
# Scaling factor: 1/sqrt(d_k) = 1/sqrt(2) ≈ 0.707
# Why scale? As d_k grows, dot products grow in magnitude, pushing softmax
# into regions with very small gradients. Scaling prevents this.
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=0)

# ============================================================================
# STEP 6: Compute Context Vector (Weighted Sum of Values)
# ============================================================================
# Use attention weights to create a weighted combination of value vectors
# attn_weights_2: (6,) - weights for each word
# values: (6, 2) - value vectors for all words
# Result: (2,) - context vector for "journey"
context_vec_2 = attn_weights_2 @ values

# ============================================================================
# OUTPUT
# ============================================================================
print("Context vector for 'journey' (2D):")
print(context_vec_2)
print("\nAttention weights (how much to attend to each word):")
print(attn_weights_2)
print("\nSum of weights (should be 1.0):")
print(attn_weights_2.sum())
