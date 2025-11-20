import os
import tensorflow as tf
import matplotlib.pyplot as plt

def load_binary_dataset(base_dir, image_size=(256, 256), batch_size=16):
    return tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        label_mode='binary',
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True
    )


dataset = load_binary_dataset("imgwaste")


IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
SEED = 123


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "imgwaste",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='rgb'
)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "imgwaste",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    color_mode='rgb'
)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


model.fit(train_ds, epochs=40, verbose=1)


loss, acc = model.evaluate(val_ds)
print(f"Dokładność: {acc:.2%}")


model.save("binary_waste_classifier.h5")
