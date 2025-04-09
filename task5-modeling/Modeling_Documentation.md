# Modeling Documentation for Segmentation Framework

This document describes the architecture, loss functions, metrics, and training utilities for semantic segmentation models used in marine debris and water body classification tasks.

---

## 📌 Model Architectures

### `attention_unet(input_shape, num_classes)`
Builds an Attention U-Net with:
- **Band Attention**: Applies channel-wise attention across input bands.
- **Skip Attention**: Uses gating mechanisms in skip connections.
- **Encoder-Decoder**: Classic U-Net structure with downsampling and upsampling blocks.

---

### `DeeplabV3Plus(input_shape, num_classes)`
Implements DeepLabV3+ with a **custom ResNet50 backbone** adapted for 9-band input.

- Uses intermediate layers from ResNet for:
  - Deep features (1/16 resolution)
  - Skip features (1/4 resolution)
- Fuses features using upsampling and concatenation
- Outputs a multi-class softmax mask

---

## 🎯 Attention Modules

### `attention_gate(x, g, inter_channels)`
Combines skip connection and upsampled decoder feature via attention.

### `band_attention(x)`
Applies learned weights to each band for importance reweighting.

---

## ⚙️ Loss Functions

### `multiclass_dice_loss(y_true, y_pred)`
- Dice loss for multi-class problems.
- Focuses on overlapping region between prediction and ground truth.

### `multiclass_focal_loss(gamma=2.0, alpha=0.25)`
- Focal loss to focus on hard-to-classify examples.
- `gamma` controls focus, `alpha` balances class weights.

### `combined_multiclass_loss(y_true, y_pred)`
- Weighted combination of dice and focal loss.

### `weighted_categorical_crossentropy_new(weights)`
- Custom cross-entropy loss with manual class weights.

### `weighted_focal_loss(weights, gamma=2.0)`
- Class-weighted variant of focal loss for unbalanced datasets.

---

## 📏 Metrics

### `multiclass_iou(y_true, y_pred)`
- Calculates intersection-over-union across all classes.

### `MulticlassF1Score(num_classes)`
- Custom TensorFlow metric for class-wise F1 scores.
- Maintains internal confusion matrix statistics.

---

## 🚀 Training Utilities

### `compile_and_train(model, train_dataset, val_dataset, save_path, epochs=50)`
- Compiles and trains a model.
- Supports different loss configurations.
- Uses `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`.

### `evaluate_and_save(model, test_dataset, model_path)`
- Evaluates a trained model.
- Saves the final Keras model to disk.
- Prints IoU and F1 score.

---

## 🧪 Inference

### `predict_mask(model, image_tensor)`
- Accepts a single image and returns predicted segmentation mask.

---

## ✅ Example Usage

```python
# Create and train model
model = attention_unet(input_shape=(256, 256, 9), num_classes=8)
compile_and_train(model, train_dataset, val_dataset, save_path="model.h5", epochs=100)

# Evaluate
evaluate_and_save(model, test_dataset, model_path="model_final.keras")

# Predict on new image
predicted_mask = predict_mask(model, sample_image)
```

---

## 📝 Notes

- Both models are designed to work with 9-band satellite inputs.
- Loss functions are tailored for class imbalance (e.g. marine debris is rare).
- Evaluation focuses on mean IoU and mean F1-score for multi-class prediction.