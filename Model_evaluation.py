# ================================
# COMPLETE MODEL EVALUATION SCRIPT
# ================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------------------------
# Paths
# -------------------------
DATA_DIR = "D:/CNN_Project/CNN_Project/data_aug_max/test"
MODEL_PATH = "D:/CNN_Project/CNN_Project/cnn_model_balanced1.h5"

img_size = (128, 128)
batch_size = 32
num_classes = 4

# -------------------------
# Step 1: Load Model
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------
# Step 2: Load Test Dataset
# -------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,      # IMPORTANT for confusion matrix
    label_mode='int'
)

class_names = test_ds.class_names
print("✅ Test dataset loaded!")
print("Classes:", class_names)

# -------------------------
# Step 3: Evaluate Accuracy & Loss
# -------------------------
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"\n📊 Test Accuracy: {accuracy:.4f}")
print(f"📉 Test Loss: {loss:.4f}")

# -------------------------
# Step 4: Get Predictions
# -------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------------------------
# Step 5: Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# Step 6: Classification Report
# -------------------------
print("\n📄 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))