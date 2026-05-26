
# makemore

A small experiment that learns character-level bigram statistics from a list of names and uses a tiny PyTorch model to generate new names. The core work is in the Jupyter notebook `makemore.ipynb` and the training data is `names.txt`.

**Status:** Research / learning code — lightweight, educational, and not intended for production use.

**Contents**
- `makemore.ipynb` — Jupyter notebook with data prep, visualization, a simple bigram model and a tiny neural training loop.
- `names.txt` — dataset of example names used to build transition statistics.

**What it does**
- Builds a character-to-character (bigram) transition matrix from `names.txt`.
- Normalizes counts into probabilities and samples new names from that distribution.
- Demonstrates a minimal trainable weight matrix implementation in PyTorch that learns to predict the next character.

Requirements
- Python 3.8+
- PyTorch (CPU or GPU build) — e.g. `pip install torch`
- Matplotlib (for the visualization) — `pip install matplotlib`
- Jupyter (recommended) — `pip install jupyterlab`

Quickstart
1. Install dependencies (example):

```bash
python -m pip install --upgrade pip
python -m pip install torch matplotlib jupyterlab
```

2. Open the notebook

```bash
jupyter lab makemore.ipynb
```

3. In the notebook, run cells in order. The notebook walks through:
- Loading `names.txt`
- Constructing count matrix `N` and probability matrix `P`
- Visualizing bigram counts
- Sampling names from `P`
- Setting up a small PyTorch weight matrix `W`, training it, and sampling from the learned model

Usage (headless sampling snippet)

If you want to run sampling outside the notebook, this minimal snippet reproduces the sampling logic used in the notebook:

```python
# minimal sampling example
words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build counts
import torch
N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
for w in words:
	chs = ['.'] + list(w) + ['.']
	for a,b in zip(chs, chs[1:]):
		N[stoi[a], stoi[b]] += 1

P = (N + 1).float()
P = P / P.sum(1, keepdims=True)

# sample
g = torch.Generator().manual_seed(42)
for _ in range(10):
	ix = 0
	out = []
	while True:
		p = P[ix]
		ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
		if ix == 0:
			break
		out.append(itos[ix])
	print(''.join(out))
```

Notes and ideas
- The notebook illustrates the difference between using raw counts, normalized probabilities, and learning a parameterized matrix `W` with gradient descent.
- You can extend this by using higher-order n-grams, RNNs/Transformers, temperature sampling, or by augmenting the dataset.

Contributing
- This repository is intended as an educational sandbox. Feel free to open issues or PRs with improvements or experiments.

License
- This repository does not include a license file. Add a license if you want to explicitly permit reuse.

