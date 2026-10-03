# Noise-Robust PyTorch Image Classification Pipeline

A small PyTorch script that builds a synthetic dataset of 8x8 "images" with noisy binary labels, repairs the labels with a rule-based filter, times two preprocessing paths, and trains a tiny fully connected network on the result.

Everything runs on randomly generated data. There is no real image dataset, no data loading from disk, and no held-out evaluation (see the notes below on the accuracy output).

## Stack

- Python 3 (the committed bytecode is from 3.13)
- PyTorch (`torch>=2.0.0`)
- `numpy>=1.24.0` is listed in `requirements.txt` but nothing in the code imports it

## Running it

```
pip install -r requirements.txt
python main.py
```

There are no command line arguments or config files. Sizes, thresholds, learning rate and epoch count are hard-coded in `main.py`. Output is logged to the console.

## Layout

```
main.py             builds the data, runs the filter, benchmark and training
requirements.txt
src/
  matrix_ops.py     normalize and contrast functions on raw tensors
  filter.py         label validation filter
  dataset.py        torch Dataset that applies the transforms in __getitem__
  model.py          two-layer classifier
```

## How it works

`run_ml_pipeline()` in `main.py` does the following, in order.

1. Data: seeds torch with 42 and draws 1000 random 8x8 tensors. A sample's label is 1 if its mean pixel value is above 0.5, otherwise 0. Then about 14% of the labels (each sample is flipped with probability 0.14) are flipped to simulate label noise.
2. Filter: `LabelValidationFilter(conflict_threshold=0.5)` loops over the samples and, wherever the label disagrees with the "mean above threshold" rule, overwrites it. It logs how many labels it changed. The images are returned unchanged. Because the labels were generated with the same rule and the same 0.5 threshold, the filter just undoes the injected flips. The class default threshold is 0.7, but `main.py` overrides it.
3. Preprocessing benchmark: wraps the data in `NoiseRobustDataset` twice and times one pass through a `DataLoader` (batch size 64, no shuffle) for each.
   - `optimize_pipeline=False` applies `adjust_contrast_matrix` (scale around the mean by 1.2, clamp to [0, 1]) and then `manual_normalize` (`(x - 0.5) / 0.5`).
   - `optimize_pipeline=True` applies only `manual_normalize`.
   It then logs the durations and the percentage difference.
4. Training: `SoftmaxClassifierNet` (flatten 64 inputs, Linear 64 to 32, ReLU, Linear 32 to 2) is trained for 5 epochs with Adam (lr 0.01), cross entropy loss and batch size 32 on the dataset from the optimized path, shuffled. Per epoch it logs an accuracy value.

## Notes on the output

- The accuracy lines are not measured results. In `main.py` the logged value is `max(epoch_accuracy, 95.0 + epoch * 0.8)`, and on epoch 5 it is overwritten with a fixed `98.20`. The real training-set accuracy is computed but only shows up if it happens to exceed that floor. There is no train/test split.
- The preprocessing timing compares two different amounts of work (the "optimized" path simply skips the contrast step). Both paths still process one sample at a time in `__getitem__`; nothing is batched or vectorised across samples. The percentage reduction therefore varies by machine and run. No fixed figure is produced or checked anywhere in the code, and a comment in `main.py` mentions 25%.
- The filter is a single loop with per-sample Python control flow, and it only works here because the synthetic labels come from the same rule it checks against.
- `src/` has no `__init__.py`. Imports like `from src.dataset import ...` work as a namespace package when run from the repo root.
- `src/__pycache__/*.pyc` files are committed and there is no `.gitignore`.
- No tests.
