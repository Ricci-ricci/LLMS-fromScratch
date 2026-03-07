import tiktoken

tokeniser = tiktoken.get_encoding("gpt2")


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokeniser.encode(raw_text)
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]

y = enc_sample[1 : context_size + 1]

print(f"x: {x}")
print(f"y:    {y}")


for i in range(1, context_size + 1):
    context = enc_sample[
        :i
    ]  # slicing from the start of the sample to the current index i not including i   exemple [0 , 1 , 2, 3 , 4] and i slice [:1] will give [0]  and slice [:2] will give [0, 1] and so on
    desired = enc_sample[
        i
    ]  # will return the token at index i, which is the next token in the sequence that we want to predict based on the context
    contextText = tokeniser.decode(context)
    desiredText = tokeniser.decode([desired])
    print(f" {contextText} ----->  {desiredText}")
