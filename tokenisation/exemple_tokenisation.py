import re  # Regular expressions for text splitting

# ============================================================================
# BASIC TOKENIZATION EXAMPLE
# ============================================================================
# This demonstrates the fundamental steps to tokenize text:
# 1. Split text into tokens (words and punctuation)
# 2. Build a vocabulary (unique tokens)
# 3. Map each token to a unique integer ID

# ============================================================================
# STEP 1: Read the text file
# ============================================================================
# Open and read the entire file content as a single string
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ============================================================================
# STEP 2: Split text into tokens using regex
# ============================================================================
# re.split with capturing groups (parentheses) keeps the delimiters
# Pattern explanation:
# - [,.:;?_!"()\'] : Any of these punctuation characters
# - -- : Double dash (em-dash)
# - \s : Any whitespace (spaces, tabs, newlines)
# The | operator means "OR" between these patterns
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

# ============================================================================
# STEP 3: Clean up the tokens
# ============================================================================
# Remove empty strings and strip whitespace from each token
# List comprehension with filter:
# - item.strip() removes leading/trailing whitespace
# - if item.strip() filters out empty strings (they're "falsy")
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# ============================================================================
# STEP 4: Build vocabulary from unique tokens
# ============================================================================
# set(preprocessed) removes duplicates (only unique tokens)
# sorted() sorts alphabetically for consistent ordering
all_word = sorted(set(preprocessed))

# Count total unique tokens in vocabulary
vocab_size = len(all_word)

# ============================================================================
# STEP 5: Create token-to-ID mapping
# ============================================================================
# enumerate(all_word) creates (index, token) pairs: (0, "a"), (1, "and"), ...
enum_vocab = enumerate(all_word)

# Dictionary comprehension: {token: id}
# Swaps the (index, token) pairs to create {token: index} mapping
vocab = {token: integer for integer, token in enum_vocab}

# ============================================================================
# STEP 6: Display first 50 vocabulary entries
# ============================================================================
# Print the first 51 entries (0 to 50) of the vocabulary
for i, items in enumerate(vocab.items()):
    print(items)  # items is a (token, id) tuple
    if i > 50:
        break
