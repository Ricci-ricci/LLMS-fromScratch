import torch

# this is used for exemple purposes, in practice, you would use the tokeniser to convert your text into embeddings, and then use those embeddings as inputs to the attention mechanism.
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your(x^1)
        [0.55, 0.87, 0.66],  # journey(x^2)
        [0.57, 0.85, 0.64],  # starts(x^3)
        [0.22, 0.58, 0.33],  # with(x^4)
        [0.77, 0.25, 0.10],  # one(x^5)
        [0.05, 0.80, 0.55],
    ]  # step(x^6)
)
query = inputs[1]  # journey(x^2)
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_naive = softmax_naive(attn_scores_2)


context_vector_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_2 += attn_weights_naive[i] * x_i
print(context_vector_2)


attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

attn_weights = torch.softmax(attn_scores, dim=1)

all_context_vecs = attn_weights @ inputs

print(all_context_vecs)
print(context_vector_2)
