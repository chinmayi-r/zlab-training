# Title: When Regularization Backfires in Small-Data Image Classification

## Keywords
regularization, small data, image classification, negative results, overfitting

## TL;DR
Common regularization tricks are supposed to help when data is scarce, but
sometimes they hurt. Investigate when and why this happens.

## Abstract
Practitioners reach for regularization (dropout, weight decay, data augmentation,
label smoothing) almost reflexively when training on small datasets, on the
assumption that these techniques curb overfitting and improve generalization.
Yet in practice the picture is murkier: some regularizers interact badly with
small-sample regimes, with each other, or with particular architectures, and can
leave a model worse off than a plain baseline. This work explores the conditions
under which standard regularization backfires in low-data image classification.
We aim to characterize the failure modes, identify which combinations are most
fragile, and surface practical guidance about when a regularizer is likely to
help versus hurt. The intended contribution is a clearer, empirically grounded
account of a phenomenon practitioners encounter but rarely report.
