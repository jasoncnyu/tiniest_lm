# Tiny Word-Level Language Model (NumPy)

This is a minimal word-level language model implemented entirely with NumPy — no deep learning frameworks or GPUs required.

It trains on a simple `.txt` file (like lyrics or poems) to predict the next word given a fixed-size word context.  
The model uses trainable word embeddings and mean pooling as a context representation.

> 🧠 This is a tiny educational example that captures the core mechanics of Large Language Models (LLMs) — just without the "large".

---

## 🔧 Features

- Word-level training using only **NumPy**
- Runs entirely on **CPU** — no GPU or CUDA needed
- Trains embeddings and softmax projection matrix
- Predicts next word using mean context embedding
- Includes simple text generation
- Zero external ML frameworks (no PyTorch, TensorFlow)
- Great for **understanding how LLMs work at a fundamental level**

---

## 📁 Files

- `tiniest_lm.py` – main script containing training and generation logic  
- `lyrics.txt` – training data (you can replace this with any plain text)

---

## 📌 Example Training and Generation Log

After training for 100,000 steps, the model starts to produce accurate predictions:

```
Step 0000 | Input: and i was crying | Target: on | Pred: alone | Loss: 4.9488
Step 10000 | Input: air see the lights | Target: see | Pred: the | Loss: 1.5215
Step 20000 | Input: somewhere we can be | Target: alone | Pred: we | Loss: 0.1785
Step 30000 | Input: eyes and the flashback | Target: starts | Pred: out | Loss: 0.3747
Step 40000 | Input: somewhere we can be | Target: alone | Pred: out | Loss: 0.0218
Step 50000 | Input: knelt to the ground | Target: and | Pred: ground | Loss: 0.0570
Step 60000 | Input: in summer air see | Target: the | Pred: how | Loss: 0.0056
Step 70000 | Input: know i talked to | Target: your | Pred: think | Loss: 0.0263
Step 80000 | Input: romeo take me somewhere | Target: we | Pred: are | Loss: 0.0075
Step 90000 | Input: juliet but you were | Target: everything | Pred: but | Loss: 0.0478

Generated:
romeo take me somewhere we can be alone i will be waiting you all
```

## 🚀 Usage

### 1. Prepare training text
Place your plain-text training file as `lyrics.txt` in the same directory.  
Use space-separated words (punctuation optional).

### 2. Run training and generation
```bash
python tiniest_lm.py
```

The model trains on CPU in about 10 seconds — no GPU required.

---

## 📎 Notes

- This project is intended for **educational or experimental purposes**.
- Performance is not optimized — training takes several minutes on a typical CPU.
- You can replace `lyrics.txt` with your own text file to train on different data.

---

## ☕ Support

If you find this project helpful, you can [buy me a coffee](https://www.buymeacoffee.com/jcny) 🙏

---

## 📜 License

MIT License
