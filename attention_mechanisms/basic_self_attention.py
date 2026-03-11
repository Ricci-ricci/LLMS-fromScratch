import torch  # PyTorch for tensor operations

# ============================================================================
# SELF-ATTENTION MECHANISM - BASIC IMPLEMENTATION
# ============================================================================
# This demonstrates the core concepts of self-attention used in Transformers.
# In practice, you would use a tokenizer to convert text into embeddings first.

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
# PART 1: Computing Attention for a Single Query Word
# ============================================================================
# Let's compute attention for the word "journey" (index 1)

query = inputs[1]  # Select "journey" as our query: [0.55, 0.87, 0.66]

# Step 1: Calculate attention scores (similarity between query and all words)
# Create empty tensor to store scores for all 6 words
attn_scores_2 = torch.empty(inputs.shape[0])

# Compute dot product between query and each word embedding
# Dot product measures similarity: higher value = more similar
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)  # Similarity score

# Step 2: Normalize scores (naive approach - simple division by sum)
# ⚠️ This is NOT the best way - softmax is better (see below)
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()


# ============================================================================
# Better Normalization: Softmax Function
# ============================================================================
def softmax_naive(x):
    """
    Softmax converts raw scores into probabilities that sum to 1.

    Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)

    Why softmax instead of simple division?
    - Handles negative numbers properly
    - Exponential amplifies differences (more discriminative)
    - Always produces valid probability distribution

    Args:
        x: Tensor of raw scores

    Returns:
        Tensor of probabilities (all positive, sum to 1)
    """
    return torch.exp(x) / torch.exp(x).sum(dim=0)


# Step 3: Apply softmax to get proper attention weights (probabilities)
attn_weights_naive = softmax_naive(attn_scores_2)
# Now each weight represents "how much attention to pay" to each word
# Example: [0.15, 0.30, 0.25, 0.10, 0.15, 0.05] (sums to 1.0)

# Step 4: Compute context vector (weighted sum of all word embeddings)
# This creates a new representation that blends information from all words
context_vector_2 = torch.zeros(query.shape)  # Initialize as [0.0, 0.0, 0.0]

for i, x_i in enumerate(inputs):
    # Multiply each word embedding by its attention weight and accumulate
    # High weight = more influence in the final context vector
    context_vector_2 += attn_weights_naive[i] * x_i

print("Context vector for 'journey':")
print(context_vector_2)


# ============================================================================
# PART 2: Computing Attention for ALL Words at Once (Full Self-Attention)
# ============================================================================
# Instead of computing attention for just one word, compute for all words simultaneously
# This is much more efficient and is how it's done in practice

# Step 1: Compute all pairwise attention scores
# Create a 6x6 matrix where entry [i,j] = similarity between word i and word j
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):  # For each query word
    for j, x_j in enumerate(inputs):  # Compare with each key word
        # Compute similarity between word i and word j
        attn_scores[i, j] = torch.dot(x_i, x_j)

# Result: A symmetric matrix showing all word-to-word similarities
# Each row represents attention scores for one query word

# Step 2: Apply softmax to each row to get attention weights
# dim=1 means apply softmax across columns (for each row independently)
# Each row becomes a probability distribution over all words
attn_weights = torch.softmax(attn_scores, dim=1)

# Step 3: Compute context vectors for all words using matrix multiplication
# This single operation computes weighted sums for all 6 words at once!
# attn_weights: (6, 6) - attention weights
# inputs: (6, 3) - word embeddings
# Result: (6, 3) - context vectors for all 6 words
all_context_vecs = attn_weights @ inputs

# ============================================================================
# Verification: Compare single-word and full-attention results
# ============================================================================
print("\nAll context vectors (6 words):")
print(all_context_vecs)

print("\nSingle context vector for 'journey' (should match row 1 above):")
print(context_vector_2)
# Note: all_context_vecs[1] should be very similar to context_vector_2
