import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from skimage.feature import hog

IMG_SIZE = 64

def load_data(data_dir="data/images"):
    images, labels = [], []
    for label, name in enumerate(["cat", "dog"]):
        path = os.path.join(data_dir, name)
        for fname in os.listdir(path):
            img = cv2.imread(os.path.join(path, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                   feature_vector=True)
        features.append(feat)
    return np.array(features)

print("Loading images...")
X, y = load_data()
print(f"Loaded {len(X)} images ({sum(y==0)} cats, {sum(y==1)} dogs)")

print("Extracting HOG features...")
X_feat = extract_hog_features(X)
print(f"Feature dimension per image: {X_feat.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.2, random_state=42, stratify=y
)

print("Training SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Cat", "Dog"],
                                        cmap="Blues")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.savefig("outputs/confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved -> outputs/confusion_matrix.png")

# Sample predictions
sample_images, sample_labels = X[:10], y[:10]
sample_feats = extract_hog_features(sample_images)
sample_preds = svm.predict(sample_feats)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i])
    true_label = "Cat" if sample_labels[i] == 0 else "Dog"
    pred_label = "Cat" if sample_preds[i] == 0 else "Dog"
    color = "green" if sample_labels[i] == sample_preds[i] else "red"
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_predictions.png", dpi=150)
print("Sample predictions saved -> outputs/sample_predictions.png")
