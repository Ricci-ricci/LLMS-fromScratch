import re  # Regular expressions for text splitting

# ============================================================================
# SIMPLE TOKENIZER V1
# ============================================================================
# Basic tokenizer that converts text to integer IDs and back.
# This version does NOT handle unknown tokens - it will crash if it encounters
# a word that wasn't in the training vocabulary.


class simpleTokeniseurV1:
    def __init__(self, vocab):
        """
        Initialize the tokenizer with a vocabulary.

        Args:
            vocab: Dictionary mapping token strings to integer IDs
                   Example: {"hello": 0, "world": 1, ",": 2}
        """
        # Store the vocab for encoding (token → id)
        self.str_to_int = vocab
        # Create reverse mapping for decoding (id → token)
        # Dict comprehension that swaps keys and values
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        Convert text into a list of token IDs.

        Args:
            text: String to encode

        Returns:
            List of integer token IDs
        """
        # Split text on punctuation and whitespace, keeping delimiters
        # The parentheses () in regex make it a capturing group (keeps delimiters)
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        # Remove empty strings and strip whitespace
        # List comprehension with filter: only keep non-empty items after stripping
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Convert each token to its ID using the vocabulary
        # ⚠️ WARNING: This will crash with KeyError if token not in vocab!
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        Convert a list of token IDs back into text.

        Args:
            ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        # Convert each ID back to its token string and join with spaces
        text = " ".join([self.int_to_str[i] for i in ids])

        # Fix spacing around punctuation using regex substitution
        # \s+ matches one or more spaces
        # ([,.?!"()\']) captures punctuation marks
        # r"\1" replaces with just the punctuation (removes the space before it)
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)

        return text


# ============================================================================
# SIMPLE TOKENIZER V2 (with Unknown Token Handling)
# ============================================================================
# Improved version that handles unknown tokens gracefully using <|unk|>


class simpleTokeniseurV2:
    def __init__(self, vocab):
        """
        Initialize the tokenizer with a vocabulary.

        Args:
            vocab: Dictionary mapping token strings to integer IDs
                   Should include special tokens like <|unk|> and <|endoftext|>
        """
        # Store the vocab for encoding (token → id)
        self.str_to_int = vocab
        # Create reverse mapping for decoding (id → token)
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        Convert text into a list of token IDs, handling unknown tokens.

        Args:
            text: String to encode

        Returns:
            List of integer token IDs
        """
        # Split text on punctuation and whitespace, keeping delimiters
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        # Remove empty strings and strip whitespace
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Handle unknown tokens: replace with <|unk|> if not in vocabulary
        # Conditional expression (ternary): value_if_true if condition else value_if_false
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        # Convert each token to its ID (all tokens are guaranteed to be in vocab now)
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        Convert a list of token IDs back into text.

        Args:
            ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        # Convert each ID back to its token string and join with spaces
        text = " ".join([self.int_to_str[i] for i in ids])

        # Fix spacing around punctuation using regex substitution
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)

        return text


# ============================================================================
# EXAMPLE USAGE: Building a vocabulary and using V1
# ============================================================================

# Read the text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Preprocess: split on punctuation and whitespace
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Build vocabulary from unique tokens (sorted alphabetically)
all_word = sorted(set(preprocessed))  # set() removes duplicates, sorted() orders them
vocab_size = len(all_word)  # Total number of unique tokens

# Create token → id mapping using enumerate
# enumerate gives us (index, value) pairs: (0, "a"), (1, "and"), (2, "apple"), ...
enum_vocab = enumerate(all_word)
vocab = {token: integer for integer, token in enum_vocab}

# Create tokenizer with the vocabulary
tokenizer = simpleTokeniseurV1(vocab)

# Test encoding
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
pride."""
ids = tokenizer.encode(text)

# ============================================================================
# EXAMPLE USAGE: Building vocabulary with special tokens for V2
# ============================================================================

text = "Hello, do you like tea?"

# Build vocabulary with special tokens
# Special tokens handle edge cases:
# - <|unk|>: Unknown tokens (words not seen during training)
# - <|endoftext|>: Marks the end of a document/sequence
all_tokens = sorted(list(set(preprocessed)))  # Get unique tokens, sorted
all_tokens.extend(["<|endoftext|>", "<|unk|>"])  # Add special tokens at the end
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# ============================================================================
# EXAMPLE: Using V2 with special tokens
# ============================================================================

# Create two separate texts
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

# Join them with the special <|endoftext|> separator
# This tells the model "text1 ends here, text2 starts here"
text3 = " <|endoftext|> ".join((text1, text2))

# Create V2 tokenizer and test encode/decode
tokenize2 = simpleTokeniseurV2(vocab)
ids = tokenize2.encode(text3)  # Convert to IDs
token2 = tokenize2.decode(ids)  # Convert back to text

# Print results
print(ids)  # List of integer IDs
print(token2)  # Reconstructed text
