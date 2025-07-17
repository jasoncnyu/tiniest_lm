import numpy as np

with open("lyrics.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = text.strip().split()
vocab = sorted(set(words))
vocab_size = len(vocab)

block_size = 4
hidden_size = 8
lr = 0.05

W_embed = np.random.normal(0, 0.01, size=(vocab_size, hidden_size))
W_proj = np.random.normal(0, 0.01, size=(hidden_size, vocab_size))

for step in range(100000):
    ix = np.random.randint(0, len(words) - block_size - 1)
    x = words[ix : ix + block_size]
    y = words[ix + block_size]

    x_indices = [vocab.index(w) for w in x]
    y_index = vocab.index(y)

    emb = W_embed[x_indices]  # shape: (block_size, hidden_size)
    context = emb.mean(axis=0)
    logits = context @ W_proj
    probs = np.exp(logits) / np.exp(logits).sum()

    loss = -np.log(probs[y_index])

    probs[y_index] -= 1
    dW_proj = np.outer(context, probs)
    dcontext = probs @ W_proj.T

    for idx in x_indices:
        W_embed[idx] -= lr * dcontext
    W_proj -= lr * dW_proj

    if step % 10000 == 0:
        pred_idx = np.argmax(probs)
        pred_word = vocab[pred_idx]
        print(f"Step {step:04d} | Input: {' '.join(x)} | Target: {y} | Pred: {pred_word} | Loss: {loss:.4f}")

def generate(start, length=8):
    out = start.strip().split()
    for _ in range(length):
        context = out[-block_size:]
        context_indices = [vocab.index(w) for w in context]
        emb = W_embed[context_indices]
        context_vec = emb.mean(axis=0)
        logits = context_vec @ W_proj
        probs = np.exp(logits) / np.exp(logits).sum()
        next_idx = np.random.choice(vocab_size, p=probs)
        out.append(vocab[next_idx])
    return ' '.join(out)

print("\nGenerated:")
print(generate("romeo take me somewhere", length=10))
