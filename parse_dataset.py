import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random

INPUT_SIZE = (480, 720)
AUGMENTATIONS_PER_IMAGE = 20
IMAGES_DIR = "dataset/manual_labeled/images"
ANNOTATIONS_PATH = "dataset/manual_labeled/result.json"
OUTPUT_DIR = "tfrecord_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_coco_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data["images"]
    annotations = data["annotations"]
    ann_map = {}
    for ann in annotations:
        image_id = ann["image_id"]
        polygon = ann["segmentation"][0]
        if len(polygon) < 8:
            print(f"Warning: Polygon for image_id {image_id} has {len(polygon)} coordinates, expected at least 8. Skipping.")
            continue
        polygon = polygon[:8]
        ann_map[image_id] = polygon
    return images, ann_map

def random_transform(image, polygon):
    h, w = image.shape[:2]
    scale = random.uniform(0.5, 1)
    nh, nw = int(h * scale), int(w * scale)
    image = cv2.resize(image, (nw, nh))
    polygon = np.array(polygon).reshape(-1, 2)
    polygon = polygon * scale
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (nw, nh))
    ones = np.ones((polygon.shape[0], 1))
    points_ones = np.hstack([polygon, ones])
    polygon = M.dot(points_ones.T).T
    crop_w, crop_h = INPUT_SIZE[1], INPUT_SIZE[0]
    cx = nw // 2 + random.randint(-nw // 10, nw // 10)
    cy = nh // 2 + random.randint(-nh // 10, nh // 10)
    x0 = max(cx - crop_w // 2, 0)
    y0 = max(cy - crop_h // 2, 0)
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    if x1 > nw:
        x0 = nw - crop_w
        x1 = nw
    if y1 > nh:
        y0 = nh - crop_h
        y1 = nh
    image = image[y0:y1, x0:x1]
    polygon -= np.array([x0, y0])
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, crop_w)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, crop_h)
    return image, polygon.reshape(-1).tolist()

def ensure_rgb(image):
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image

def serialize_example(image, polygon):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
        'polygon': tf.train.Feature(float_list=tf.train.FloatList(value=polygon))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def create_tfrecord(images, ann_map, out_path, indices, augment=True):
    with tf.io.TFRecordWriter(out_path) as writer:
        for idx in tqdm(indices):
            image_info = images[idx]
            filename = image_info["file_name"]
            image_path = os.path.join(IMAGES_DIR, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue
            image = ensure_rgb(image)
            h0, w0 = image.shape[:2]
            polygon = ann_map.get(image_info["id"])
            if polygon is None:
                print(f"Warning: No annotation for image id {image_info['id']}")
                continue
            if augment:
                for _ in range(AUGMENTATIONS_PER_IMAGE):
                    img_aug, poly_aug = random_transform(image, polygon)
                    img_aug = cv2.resize(img_aug, (INPUT_SIZE[1], INPUT_SIZE[0]))
                    poly_aug = np.array(poly_aug).reshape(-1, 2)
                    scale_x = INPUT_SIZE[1] / img_aug.shape[1]
                    scale_y = INPUT_SIZE[0] / img_aug.shape[0]
                    poly_aug[:, 0] *= scale_x
                    poly_aug[:, 1] *= scale_y
                    poly_aug = poly_aug.reshape(-1).tolist()
                    if len(poly_aug) != 8:
                        print(f"Warning: Augmented polygon has {len(poly_aug)} coordinates, expected 8. Skipping.")
                        continue
                    img_aug = img_aug.astype(np.uint8)
                    writer.write(serialize_example(img_aug, poly_aug))
            else:
                image_resized = cv2.resize(image, (INPUT_SIZE[1], INPUT_SIZE[0]))
                scale_x = INPUT_SIZE[1] / w0
                scale_y = INPUT_SIZE[0] / h0
                polygon = np.array(polygon).reshape(-1, 2)
                polygon[:, 0] *= scale_x
                polygon[:, 1] *= scale_y
                polygon = polygon.reshape(-1).tolist()
                image_resized = image_resized.astype(np.uint8)
                writer.write(serialize_example(image_resized, polygon))

def split_indices(images, train_ratio=0.75, val_ratio=0.20):
    n = len(images)
    idxs = list(range(n))
    random.shuffle(idxs)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))
    return idxs[:train_cut], idxs[train_cut:val_cut], idxs[val_cut:]

def main():
    images, ann_map = load_coco_annotations(ANNOTATIONS_PATH)
    train_idx, val_idx, test_idx = split_indices(images)
    print("Creating train TFRecord...")
    create_tfrecord(images, ann_map, os.path.join(OUTPUT_DIR, "train.tfrecord"), train_idx, augment=True)
    print("Creating val TFRecord...")
    create_tfrecord(images, ann_map, os.path.join(OUTPUT_DIR, "val.tfrecord"), val_idx, augment=True)
    print("Creating test TFRecord (only original images)...")
    create_tfrecord(images, ann_map, os.path.join(OUTPUT_DIR, "test.tfrecord"), test_idx, augment=False)
if __name__ == "__main__":
    main()