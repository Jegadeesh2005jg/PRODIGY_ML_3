# PRODIGY_ML_3 - SVM Image Classification (Cats vs Dogs)

SVM classifier with HOG feature extraction to classify images of cats and dogs.

## Pipeline
1. Load 150 cat + 150 dog images (64x64) from HuggingFace `microsoft/cats_vs_dogs`
2. Extract HOG (Histogram of Oriented Gradients) features
3. Train RBF-kernel SVM
4. Evaluate on held-out test set

## How to Run

```bash
pip install -r requirements.txt
python data/prepare_data.py
python cat_dog_svm.py
```

## Results
~80-85% accuracy on the test set using only HOG features + linear/RBF SVM.
