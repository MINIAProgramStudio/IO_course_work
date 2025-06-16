import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random

INPUT_SIZE = (480, 720)
AUGMENTATIONS_PER_IMAGE = 200
IMAGES_DIR = "dataset/manual_labeled/images"
ANNOTATIONS_PATH = "dataset/manual_labeled/result.json"
OUTPUT_DIR = "tfrecords"
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
    polygon = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    center = polygon.mean(axis=0)
    dists = np.linalg.norm(polygon - center, axis=1)
    max_dist = dists.max()
    min_side = min(INPUT_SIZE)
    max_scale = (min_side / 2) / max_dist if max_dist > 0 else 1.0
    scale = random.uniform(0.25, min(1.5, max_scale))
    angle = random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    ones = np.ones((polygon.shape[0], 1), dtype=np.float32)
    polygon_homo = np.hstack([polygon, ones])
    transformed = (M @ polygon_homo.T).T
    min_x, min_y = transformed.min(axis=0)
    max_x, max_y = transformed.max(axis=0)
    out_h, out_w = INPUT_SIZE
    max_tx = out_w - max_x
    min_tx = -min_x
    max_ty = out_h - max_y
    min_ty = -min_y
    tx = random.uniform(min_tx, max_tx)
    ty = random.uniform(min_ty, max_ty)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(image, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    transformed = (M @ polygon_homo.T).T
    transformed[:, 0] = np.clip(transformed[:, 0], 0, out_w - 1)
    transformed[:, 1] = np.clip(transformed[:, 1], 0, out_h - 1)

    return warped, transformed.reshape(-1).tolist()

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