# Fashion Classification

Image classification of clothing items using transfer learning with a pre-trained Xception model (TensorFlow/Keras).

## Dataset

[clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small) — 10 classes: dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt.

```bash
git clone git@github.com:alexeygrigorev/clothing-dataset-small.git
```

## Approach

1. Load a pre-trained **Xception** model (ImageNet weights) and use it as a frozen feature extractor.
2. Add a `GlobalAveragePooling2D` layer plus a dense head on top for the 10 clothing classes.
3. Train with `ImageDataGenerator` (train/validation/test splits from the dataset folders).
4. Tune hyperparameters step by step:
   - Learning rate
   - Size of the inner dense layer
   - Dropout rate
   - Data augmentation (shear, zoom, horizontal flip)
5. Increase input resolution from 150x150 to 299x299 for the final model.
6. Save checkpoints with `ModelCheckpoint` (best validation accuracy) and evaluate/predict on the test set.

## Requirements

- Python
- `tensorflow` / `keras`
- `numpy`
- `matplotlib`

## Usage

Run the notebook `Fashion_Classification.ipynb` top to bottom. It expects the cloned dataset in `./clothing-dataset-small/` with `train/`, `validation/`, and `test/` subfolders. Trained model checkpoints are saved as `.h5` files (e.g. `xception_v4_1_13_0.903.h5`), which can be reloaded with `keras.models.load_model(...)` for inference.

## Notes

Best validation accuracy achieved: **~90%** using the 299x299 input size with tuned learning rate, dropout, and augmentation.
