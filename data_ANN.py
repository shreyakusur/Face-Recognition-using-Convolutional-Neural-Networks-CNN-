import os, time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd


# PATH SETTINGS

DATASET_DIR = r"C:\Users\Shreya\PycharmProjects\DL project1"
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = (64, 64)


# LOAD DATASET

ds_train = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

ds_val = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

#  Save class names before mapping
class_names = ds_train.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalize and prefetch
AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).cache().prefetch(AUTOTUNE)
ds_val = ds_val.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).cache().prefetch(AUTOTUNE)


# BUILD ANN MODEL

model = models.Sequential([
    layers.Flatten(input_shape=(*IMG_SIZE, 3)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# TRAIN & EVALUATE

start_time = time.time()
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=2)
runtime = time.time() - start_time

val_loss, val_acc = model.evaluate(ds_val, verbose=0)

print("\n ANN Validation Accuracy: {:.2f}%".format(val_acc * 100))
print(" Runtime: {:.1f} seconds".format(runtime))


# SAVE RESULTS

results = {
    'model': 'ANN',
    'image_size': IMG_SIZE[0],
    'val_accuracy': val_acc,
    'runtime_sec': runtime
}
pd.DataFrame([results]).to_csv("ann_results.csv", index=False)
