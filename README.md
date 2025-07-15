# Coffee Roasting Network in NumPy

**A minimal two-layer neural network built from scratch with NumPy to classify coffee-roast quality.**

## Overview

This repository demonstrates how to implement a complete forward‑prop neural network pipeline in pure NumPy. We:

1. **Generate synthetic coffee-roast data** (temperature & duration) with a clear linear rule.
2. **Visualize the labeling heuristic** alongside our dataset.
3. **Normalize features** to zero mean and unit variance.
4. **Build a two-layer network** (`dense_layer`, `forward`) using sigmoid activations.
5. **Make batch predictions** and convert probabilities to binary labels.
6. **Plot probability heatmaps** and **decision boundaries** in side-by-side panels.

## Architecture & Code

All code lives in the Jupyter notebook `coffee_roasting_network_numpy.ipynb`. Key components:

* **`load_coffee_data()`**: Creates a reproducible dataset using NumPy’s random generator.
* **`plot_roast()`**: Plots raw “good” vs “bad” roasts and overlays the linear boundary:
  $d \le -\tfrac{3}{85}t + 21,\;t>175,\;12<d<15$.
* **Normalization**: Compute `means` & `stds` from training data, then standardize both train and test sets.
* **`dense_layer(A_in, W, b, activation)`**: Implements $Z = A_{in}\cdot W + b$, then applies the element‑wise activation.
* **`forward(x, W1, b1, W2, b2)`**: Chains two dense layers to produce a single output.
* **`predict_probs(X, W1, b1, W2, b2)`**: Loops over examples to return an $m\times1$ array of sigmoid outputs.
* **`plot_network(X, Y, prob_fn, label_fn)`**: Draws the two‑panel visualization with probability shading and final decisions.

## Results

* The network reproduces the synthetic linear boundary with 100% accuracy on training data.
* The heatmap panel reveals the smooth transition zone of the sigmoid around the cutoff.
* The decision panel shows perfect separation of “good roast” (×) vs “bad roast” (○).

## Next Steps

* **Backpropagation**: Add gradient computation and train entirely in NumPy.
* **Vectorization**: Replace per‑example loops with matrix operations for speed.
* **Extensions**: Try ReLU, more layers or regularization, and compare with scikit‑learn’s logistic regression.

## Repository Contents

```bash
coffee-roasting-network-numpy/
├─ coffee_roasting_network_numpy.ipynb  # Main notebook
├─ README.md                            # This file
```

## ⚙️ Installation & Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/Christ02/coffee-roasting-network-numpy.git
   cd coffee-roasting-network-numpy
   ```
2. Install dependencies:

   ```bash
   pip install numpy matplotlib
   ```
3. Launch Jupyter Lab or Notebook:

   ```bash
   jupyter lab   # or jupyter notebook
   ```
4. Open and run `coffee_roasting_network_numpy.ipynb`.

---

*Crafted with ☕ and NumPy.*
