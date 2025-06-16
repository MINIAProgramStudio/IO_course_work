import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Параметри
BATCH_SIZE = 16
EPOCHS = 10
INPUT_SIZE = (480, 720, 3)
TFRECORD_DIR = "tfrecords"
TRAIN_FILE = os.path.join(TFRECORD_DIR, "train.tfrecord")
VAL_FILE = os.path.join(TFRECORD_DIR, "val.tfrecord")

class CustomPreprocessingLayerHSV(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomPreprocessingLayerHSV, self).__init__(**kwargs)

    def call(self, inputs):
        # Перетворення з RGB в HSV
        hsv_inputs = tf.image.rgb_to_hsv(inputs)

        # Застосування умов для HSV
        condition = tf.logical_and(
            tf.logical_and(
                tf.abs(hsv_inputs[..., 1] - 0.382) < 0.2,  # Умова для S (насиченість)
                tf.abs(hsv_inputs[..., 0] - 0.05) < 0.1    # Умова для H (тон)
            ),
            hsv_inputs[..., 2] < tf.reduce_mean(hsv_inputs[2])           # Умова для V (яскравість)
        )

        # Застосування правила: якщо умова виконується, піксель = [1, 1, 1], інакше [0, 0, 0]
        output = tf.where(
            condition[..., tf.newaxis],
            tf.ones_like(inputs),  # [1, 1, 1]
            tf.zeros_like(inputs)  # [0, 0, 0]
        )
        return output

    def get_config(self):
        config = super(CustomPreprocessingLayerHSV, self).get_config()
        return config




def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'polygon': tf.io.FixedLenFeature([8], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    height = parsed['height']
    width = parsed['width']
    image = tf.io.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, (height, width, 3))
    image = tf.image.resize(image, (INPUT_SIZE[0], INPUT_SIZE[1]))
    image = tf.cast(image, tf.float32) / 255.0

    polygon = parsed['polygon']  # shape (8,)
    return image, polygon

def load_dataset(tfrecord_path, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


BASE = 1
MODEL_SAVE_PATH = "models/" + "CNN_224_" + str(2**BASE) + ".keras"
def build_cnn_model(BASE):
    inputs = tf.keras.Input(shape=INPUT_SIZE)
    x = CustomPreprocessingLayerHSV()(inputs)
    x = tf.keras.layers.Conv2D(2**(BASE), 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(2**(BASE+1), 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(2**(BASE+2), 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.AveragePooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2**(BASE+3), activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(8)(x)  # 8 координат чотирикутника

    model = tf.keras.Model(inputs, outputs)
    return model

def train():
    batch_size = 16
    train_ds = load_dataset(TRAIN_FILE, batch_size=batch_size)
    val_ds = load_dataset(VAL_FILE, batch_size=batch_size)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_freq='epoch'
        )
    ]
    model = build_cnn_model(BASE)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10**4, callbacks=callbacks)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.yscale("log")
    plt.title("CNN, BASE = "+str(BASE))
    plt.show()

train()