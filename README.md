# Computer Vision: Classification and Transfer Learning

Chen Jiang explored image classification with linear/RBF SVMs, an MLP, two CNN
designs and a transfer-learning head for Dartmouth ENGS108 Applied Machine
Learning in Fall 2022. The project compares classical image features with learned
RGB representations.

## Implementation

- Grayscale, resized image vectors with linear and RBF SVM classification;
  an RBF feature-mapping experiment is retained in the student code.
- An MLP with 16- and 32-unit hidden layers.
- A three-stage 16/32/64-filter CNN with average pooling and 128/64-unit dense layers.
- A custom two-convolution CNN with max pooling and dropout.
- A student-designed pooling/dense/dropout head on the upstream frozen Xception base.

The maintained loader creates an ordered image manifest and reuses the training
class map for evaluation. Validation is reserved from training files; test order
is fixed. Standard models receive [0,1] pixels, while Xception receives its
expected [-1,1] preprocessing.

## Run

Use Python 3.12 in a virtual environment:

```sh
pip install -r requirements.txt
python -m unittest discover -v
```

[demo.ipynb](demo.ipynb) runs a small image/SVM example. For local image data,
use class-named directories under separate `train` and `test` roots.
`vision.generators(train_dir, test_dir)` creates train/validation/test iterators.
`build_model("xception", class_count, weights="imagenet")` selects pretrained
transfer learning; the default `weights=None` supports offline architecture checks.

## Checks and results

Four test methods cover stable class IDs when a test class is absent, rejection
of unknown classes, grayscale/RGBA loading, both SVM kernels, preprocessing and
disjoint splits. All four neural architectures produced normalized predictions
and completed a training batch; Xception used random weights for this offline
check. The synthetic demo ran in a fresh kernel. Full yoga-data training and
pretrained accuracy require a separate experiment.

## 2026 maintenance

AI-assisted maintenance retained the student layer structures while repairing
class alignment, evaluation ordering, Keras compatibility and transfer preprocessing.
The manifest-based loader is new support code. Selected original student
experiments remain in [historical/student_cells.py](historical/student_cells.py).
