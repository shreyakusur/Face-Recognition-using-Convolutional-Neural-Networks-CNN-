

import os, time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
import matplotlib.pyplot as plt


# DATASET PATH

DATASET_DIR = r"C:\Users\Shreya\PycharmProjects\DL project1"

BATCH_SIZE = 32
EPOCHS = 15
RESULTS = []


# CNN Model Builder

def build_cnn(input_shape, num_classes, use_dropout=False):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        # fixed dropout toggle
        layers.Dropout(0.5) if use_dropout else layers.Lambda(lambda x: x),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



# Training Loop for 3 image sizes  and for dropout variations

for size in [32, 64, 128]:
    print(f"\n--- Training for image size {size}x{size} ---")

    # Load datasets
    ds_train = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(size, size),
        batch_size=BATCH_SIZE
    )
    ds_val = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(size, size),
        batch_size=BATCH_SIZE
    )

    # get class names before prefetch
    class_names = ds_train.class_names
    num_classes = len(class_names)
    print("Detected classes:", class_names)

    # Optimize pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)

    # Train with and without dropout
    for dropout_flag in [False, True]:
        print(f"\nTraining model: image_size={size}, dropout={dropout_flag}")
        start_time = time.time()

        model = build_cnn((size, size, 3), num_classes, use_dropout=dropout_flag)
        hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=2)

        runtime = time.time() - start_time
        val_acc = hist.history['val_accuracy'][-1]

        RESULTS.append({
            'img_size': size,
            'dropout': dropout_flag,
            'val_acc': val_acc,
            'runtime_sec': runtime
        })


# Save & visualize results

df = pd.DataFrame(RESULTS)
print("\nFinal Results:\n", df)
df.to_csv("cnn_results.csv", index=False)

# Plot accuracy and runtime
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
for drop_flag in [False, True]:
    plt.plot(df[df['dropout'] == drop_flag]['img_size'],
             df[df['dropout'] == drop_flag]['val_acc'],
             marker='o', label=f"Dropout={drop_flag}")
plt.title("Validation Accuracy vs Image Size")
plt.xlabel("Image Size")
plt.ylabel("Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['img_size'], df['runtime_sec'], marker='s')
plt.title("Training Runtime vs Image Size")
plt.xlabel("Image Size")
plt.ylabel("Runtime (seconds)")

plt.tight_layout()
plt.show()
