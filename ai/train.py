import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

print("TensorFlow:", tf.__version__)
print("GPU:", bool(tf.config.list_physical_devices('GPU')))

#KONFIGURASI
DATA_DIR = '../waste_dataset'  # Sesuaikan path dataset
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
NUM_CLASSES = 6  # organic, recyclable, glass, paper, plastic, metal

#LOAD DATASET
print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_ds.class_names
print("Kelas:", class_names)

# Optimasi performa
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

#DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

#(MobileNetV2)
print("\n Building MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze awal

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#CALLBACKS
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        verbose=1
    )
]

#TRAINING
print("\nMulai training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

#EVALUASI
print("\nEvaluasi model...")

# Plot hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

#SAVE MODEL & LABELS
print("\nMenyimpan model...")

# Save model untuk training
model.save('waste_classifier.h5')

# Convert ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('mobilenetv2_waste.tflite', 'wb') as f:
    f.write(tflite_model)

# Save labels
with open('labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print("Selesai!")
print("File tersimpan:")
print("   - waste_classifier.h5 (model Keras)")
print("   - mobilenetv2_waste.tflite (model untuk HP)")
print("   - labels.txt (daftar kelas)")
print("   - training_history.png (grafik)")