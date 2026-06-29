# Title: When Regularization Backfires: A Controlled Study on CIFAR-10 with a Small CNN

## Keywords
regularization, small data, CIFAR-10, ResNet-18, negative results, compute-budget

## TL;DR
On a fixed, small CIFAR-10 subset and a fixed ResNet-18, sweep four regularizers
one at a time and measure exactly when test accuracy drops below the unregularized
baseline.

## Abstract
This study isolates when standard regularization hurts rather than helps, under a
deliberately pinned-down experimental setup so results are reproducible on a single
small GPU. CONCRETE SPECIFICATION the agent must follow:

- Dataset: CIFAR-10 (torchvision, auto-download). Use a FIXED subset of 5,000
  training images (500 per class, seed=0) to create a genuine small-data regime,
  and the full 10,000-image test set for evaluation.
- Model: ResNet-18 from torchvision (num_classes=10), trained from scratch. Do NOT
  use pretrained weights. This is the only architecture.
- Metric: top-1 test accuracy (primary). Report mean +/- std over 3 seeds (0,1,2).
- Conditions (one-variable-at-a-time vs. a no-regularization baseline):
  (1) baseline, (2) dropout p=0.3, (3) weight decay 5e-4, (4) label smoothing 0.1,
  (5) RandAugment. Exactly these five arms.
- Training budget: SGD, lr=0.1 cosine-decayed, batch size 128, 30 epochs, no early
  stopping. This is the entire compute budget -- do not grid-search hyperparameters.
- Compute constraint: must fit and finish on a single ~10GB GPU (MIG A100 slice).
  Keep batch size <= 128 and the model at ResNet-18 scale; if memory is tight,
  reduce batch size to 64 rather than changing the model or dataset.
- Deliverable: a table of test accuracy per arm and per seed, plus a bar chart with
  error bars; identify which arms fall below baseline and by how much.

The contribution is a clean, fully specified negative-results study: which of these
five standard regularizers degrade small-data CIFAR-10 accuracy under an identical,
modest training budget.
