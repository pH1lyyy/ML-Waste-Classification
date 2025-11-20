import os
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub
import shutil



path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")

project_dataset_dir = "garbage-classification-v2"


if not os.path.exists(project_dataset_dir):
    shutil.copytree(path, project_dataset_dir)



path = project_dataset_dir

if not os.path.exists("WasteClassificationNeuralNetwork"):
    os.system("git clone https://github.com/cardstdani/WasteClassificationNeuralNetwork.git")



def load_dataset_from_dir(base_dir, image_size=(256,256), batch_size=16):
    if not os.path.exists(base_dir):
        return None
    return tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb'
    )



waste_images = load_dataset_from_dir("WasteClassificationNeuralNetwork/WasteImagesDataset")
garbage_v2 = load_dataset_from_dir("garbage-classification-v2/garbage-dataset")




common_classes = [
    'cardboard', 'glass', 'metal', 'paper', 'plastic',
    'organic', 'textiles', 'e-waste', 'medical', 'wood'
]

waste_map = {
    'Aluminium': 'metal',
    'Carton': 'cardboard',
    'Glass': 'glass',
    'Organic Waste': 'organic',
    'Other Plastics': 'plastic',
    'Paper and Cardboard': 'paper',
    'Plastic': 'plastic',
    'Textiles': 'textiles',
    'Wood': 'wood'
}

garbage_v2_map = {
    'Metal': 'metal',
    'Glass': 'glass',
    'Biological': 'organic',
    'Paper': 'paper',
    'Battery': 'e-waste',
    'Trash': 'medical',
    'Cardboard': 'cardboard',
    'Shoes': 'textiles',
    'Clothes': 'textiles',
    'Plastic': 'plastic'
}

print(common_classes)



def remap_labels(dataset, class_map):
    class_map = {k.lower(): v for k, v in class_map.items()}
    orig_classes = [c.lower() for c in dataset.class_names]

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(orig_classes),
            values=tf.constant([common_classes.index(class_map[c]) for c in orig_classes])
        ),
        default_value=-1
    )

    def map_fn(images, labels):
        label_names = tf.gather(orig_classes, labels)
        new_labels = table.lookup(label_names)
        return images, new_labels

    return dataset.map(map_fn)


def precision_m(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int64)
    cm = tf.math.confusion_matrix(y_true, y_pred_classes, num_classes=numClasses)
    tp = tf.cast(tf.linalg.diag_part(cm), tf.float32)
    pred_sum = tf.cast(tf.reduce_sum(cm, axis=0), tf.float32)
    precision = tf.reduce_mean(tp / (pred_sum + tf.keras.backend.epsilon()))
    return precision

# def fpr_m(y_true, y_pred):
#
#     y_pred_classes = tf.argmax(y_pred, axis=1)
#     y_true = tf.cast(y_true, tf.int64)
#     cm = tf.math.confusion_matrix(y_true, y_pred_classes, num_classes=numClasses)
#     cm = tf.cast(cm, tf.float32)
#     fp = tf.reduce_sum(cm, axis=0) - tf.linalg.diag_part(cm)
#     fn = tf.reduce_sum(cm, axis=1) - tf.linalg.diag_part(cm)
#     tp = tf.linalg.diag_part(cm)
#     tn = tf.reduce_sum(cm) - (fp + fn + tp)
#
#     fpr = tf.reduce_mean(fp / (fp + tn + tf.keras.backend.epsilon()))
#     return fpr

def recall_m(y_true, y_pred):
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int64)
    cm = tf.math.confusion_matrix(y_true, y_pred_classes, num_classes=numClasses)
    cm = tf.cast(cm, tf.float32)
    tp = tf.linalg.diag_part(cm)
    fn = tf.reduce_sum(cm, axis=1) - tp
    recall = tf.reduce_mean(tp / (tp + fn + tf.keras.backend.epsilon()))
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


waste_images_mapped = remap_labels(waste_images, waste_map)
garbage_v2_mapped = remap_labels(garbage_v2, garbage_v2_map)
train_dataset = waste_images_mapped.concatenate(garbage_v2_mapped)

train_dataset = train_dataset.shuffle(1000)

dataset_size = 0
for _ in train_dataset:
    dataset_size += 1

train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

train_ds = train_dataset.take(train_size)
test_ds = train_dataset.skip(train_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

numClasses = len(common_classes)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(numClasses, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', precision_m, recall_m, f1_m]
)



model.fit(train_ds, epochs=50, verbose=1)

test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate(test_ds, verbose=2)
print(f"\nDokładność modelu: {test_acc:.2%}")
print(f"Precyzja: {test_prec:.2%}")
print(f"Czułość: {test_rec:.2%}")
print(f"F1: {test_f1:.2%}")
    

model.save('test_waste_model.h5')
