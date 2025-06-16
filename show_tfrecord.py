import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TFRECORD_PATH = "tfrecords/train.tfrecord"
INPUT_SIZE = (480, 720)  # (height, width)
NUM_SAMPLES = 10         # скільки зображень показати

def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'polygon': tf.io.FixedLenFeature([8], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    image = tf.reshape(image, (height, width, 3))
    polygon = example['polygon']
    return image, polygon

def show_image_with_polygon(image, polygon):
    fig, ax = plt.subplots()
    ax.imshow(image)
    polygon_np = polygon.numpy().reshape(4, 2)
    patch = patches.Polygon(polygon_np, closed=True, edgecolor='lime', facecolor='none', linewidth=2)
    ax.add_patch(patch)
    for x, y in polygon_np:
        ax.plot(x, y, 'ro')
    plt.axis('off')
    plt.show()

def main():
    raw_dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    parsed_dataset = raw_dataset.map(parse_example)

    for i, (image, polygon) in enumerate(parsed_dataset.take(NUM_SAMPLES)):
        image_np = image.numpy()
        show_image_with_polygon(image_np, polygon)

if __name__ == "__main__":
    main()
